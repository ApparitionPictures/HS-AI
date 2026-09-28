-- =====================================================================
--  HSAI telemetry mod for Half Sword  (UE4SS 3.x Lua)
--
--  Every game frame this mod reads the exact fighter state that the game
--  keeps on each character (Health, Consciousness, Stamina, All Body Tonus,
--  DED, Team Int, position, rotation, velocity, a few bone locations) and
--  streams one text line to the HS-AI python process through a named pipe
--  (or a file when the pipe is not available).  It also executes a small
--  set of commands written by python to a command file (heal, kill enemies,
--  spawn an enemy, time dilation, ...).
--
--  Nothing here changes gameplay unless python explicitly sends a command.
-- =====================================================================

local UEHelpers = require("UEHelpers")

local MOD = "HSAI"
local PROTO = 1

-- ---------------------------------------------------------------------
--  configuration (ue4ss\Mods\HSAI\config.txt overrides these)
-- ---------------------------------------------------------------------
local cfg = {
    pipe = "\\\\.\\pipe\\hsai_telemetry",
    dir = nil,                 -- defaults to %LOCALAPPDATA%\HSAI
    tick_ms = 6,               -- scheduling period of the async loop (game thread work happens once per frame at most)
    rescan_ms = 2000,          -- how often to rescan the object array for fighters
    max_enemies = 4,
    bones = { "head", "hand_r", "hand_l", "pelvis" },
    classes = { "Willie_BP_C" },
    file_fallback = true,
    debug = false,
}

local function log(s)
    print(string.format("[%s] %s\n", MOD, s))
end

local MOD_DIR = "ue4ss\\Mods\\" .. MOD
local function loadConfig()
    local f = io.open(MOD_DIR .. "\\config.txt", "r")
    if not f then
        MOD_DIR = "Mods\\" .. MOD          -- flat (stable 3.0.1) layout
        f = io.open(MOD_DIR .. "\\config.txt", "r")
    end
    if not f then return end
    for line in f:lines() do
        local k, v = line:match("^%s*([%w_]+)%s*=%s*(.-)%s*$")
        if k and v and v ~= "" then
            if k == "bones" or k == "classes" then
                local t = {}
                for item in v:gmatch("[^,%s]+") do t[#t + 1] = item end
                cfg[k] = t
            elseif k == "tick_ms" or k == "rescan_ms" or k == "max_enemies" then
                cfg[k] = tonumber(v) or cfg[k]
            elseif k == "file_fallback" or k == "debug" then
                cfg[k] = (v == "1" or v == "true")
            else
                cfg[k] = v
            end
        end
    end
    f:close()
end
loadConfig()

if not cfg.dir then
    local base = os.getenv("LOCALAPPDATA") or "."
    cfg.dir = base .. "\\HSAI"
end
-- the HS-AI installer creates %LOCALAPPDATA%\HSAI; if it is missing we fall back to the mod folder
local function dirWritable(d)
    local f = io.open(d .. "\\.probe", "w")
    if not f then return false end
    f:close()
    os.remove(d .. "\\.probe")
    return true
end
if not dirWritable(cfg.dir) then
    cfg.dir = MOD_DIR
end
local TELEMETRY_FILE = cfg.dir .. "\\telemetry.txt"
local CMD_FILE = cfg.dir .. "\\cmd.txt"
local STATUS_FILE = cfg.dir .. "\\mod_status.txt"

-- ---------------------------------------------------------------------
--  small helpers
-- ---------------------------------------------------------------------
local function isValid(o)
    return o ~= nil and o.IsValid ~= nil and o:IsValid()
end

local function tryGet(obj, name)
    local ok, v = pcall(function() return obj[name] end)
    if ok then return v end
    return nil
end

local function tryNum(obj, name, default)
    local v = tryGet(obj, name)
    if type(v) == "number" then return v end
    if type(v) == "boolean" then return v and 1 or 0 end
    local n = tonumber(v)
    if n then return n end
    return default
end

local function tryBool(obj, name)
    local v = tryGet(obj, name)
    if type(v) == "boolean" then return v end
    if type(v) == "number" then return v ~= 0 end
    return false
end

local function trySet(obj, name, value)
    local ok, err = pcall(function() obj[name] = value end)
    if not ok and cfg.debug then log("set " .. name .. " failed: " .. tostring(err)) end
    return ok
end

local function tryCall(obj, fname, ...)
    local args = { ... }
    local ok, res = pcall(function() return obj[fname](obj, table.unpack(args)) end)
    if ok then return true, res end
    return false, res
end

local function f1(x)
    if x == nil then return "0" end
    return string.format("%.1f", x)
end

local function f2(x)
    if x == nil then return "0" end
    return string.format("%.2f", x)
end

local function writeSmallFile(path, content)
    local f = io.open(path, "w")
    if not f then return false end
    f:write(content)
    f:close()
    return true
end

-- ---------------------------------------------------------------------
--  output channel: named pipe first, file fallback
-- ---------------------------------------------------------------------
local pipe = nil
local pipeRetryAt = 0
local lastFileWrite = 0

local function nowMs()
    return math.floor(os.clock() * 1000)
end

local function openPipe()
    local f = io.open(cfg.pipe, "wb")
    if f then
        f:setvbuf("no")
        pipe = f
        log("connected to pipe " .. cfg.pipe)
        return true
    end
    return false
end

local function emit(line)
    if pipe == nil and nowMs() >= pipeRetryAt then
        if not openPipe() then pipeRetryAt = nowMs() + 1000 end
    end
    if pipe ~= nil then
        local ok = pipe:write(line, "\n")
        if not ok then
            pcall(function() pipe:close() end)
            pipe = nil
            pipeRetryAt = nowMs() + 500
            log("pipe write failed, will reconnect")
        else
            return
        end
    end
    if cfg.file_fallback then
        -- in-place write, python validates the END marker and sequence number
        writeSmallFile(TELEMETRY_FILE, line .. "\n")
    end
end

-- ---------------------------------------------------------------------
--  world / fighter cache
-- ---------------------------------------------------------------------
local seq = 0
local cache = {
    pc = nil,
    fighters = {},        -- list of { obj=..., mesh=..., bones={name=FName} }
    lastRescan = -1e9,
    mapName = "",
    spawned = {},
    meshProbed = {},
    forceRescan = true,
}

local function getPC()
    if isValid(cache.pc) then return cache.pc end
    local ok, pc = pcall(function() return UEHelpers.GetPlayerController() end)
    if ok and isValid(pc) then
        cache.pc = pc
        return pc
    end
    return nil
end

local function currentMapName()
    local ok, name = pcall(function()
        local world = UEHelpers.GetWorld()
        if isValid(world) then
            local full = world:GetFullName()
            -- "World /Game/Maps/Alley.Alley" -> "Alley"
            local short = full:match("([%w_%-]+)$") or full
            return short
        end
        return ""
    end)
    if ok and name then return name end
    return ""
end

local function looksLikeFighter(obj)
    if not isValid(obj) then return false end
    local h = tryGet(obj, "Health")
    return type(h) == "number"
end

local function rescanFighters()
    local found = {}
    local seen = {}
    for _, cls in ipairs(cfg.classes) do
        local ok, list = pcall(function() return FindAllOf(cls) end)
        if ok and list then
            for _, obj in ipairs(list) do
                if looksLikeFighter(obj) then
                    local key = obj:GetAddress()
                    if not seen[key] then
                        seen[key] = true
                        found[#found + 1] = { obj = obj }
                    end
                end
            end
        end
    end
    if #found == 0 then
        -- fallback for renamed classes: any Character that has a Health number
        local ok, list = pcall(function() return FindAllOf("Character") end)
        if ok and list then
            for _, obj in ipairs(list) do
                if looksLikeFighter(obj) then
                    local key = obj:GetAddress()
                    if not seen[key] then
                        seen[key] = true
                        found[#found + 1] = { obj = obj }
                    end
                end
            end
        end
    end
    -- resolve meshes / bones once per fighter
    for _, f in ipairs(found) do
        local mesh = tryGet(f.obj, "Mesh")
        if isValid(mesh) then
            f.mesh = mesh
            f.bones = {}
            for _, b in ipairs(cfg.bones) do
                local ok, exists = tryCall(mesh, "DoesSocketExist", FName(b))
                if ok and exists then f.bones[b] = FName(b) end
            end
        end
    end
    cache.fighters = found
    cache.lastRescan = nowMs()
    cache.mapName = currentMapName()
    cache.forceRescan = false
    if cfg.debug then log(string.format("rescan: %d fighters on map %s", #found, cache.mapName)) end
end

local function vec(v)
    if v == nil then return 0, 0, 0 end
    return v.X or 0, v.Y or 0, v.Z or 0
end

local function fighterFields(f, out)
    local o = f.obj
    out[#out + 1] = f1(tryNum(o, "Health", 0))
    out[#out + 1] = f1(tryNum(o, "Consciousness", 0))
    out[#out + 1] = f1(tryNum(o, "Stamina", 0))
    out[#out + 1] = f1(tryNum(o, "All Body Tonus", 0))
    out[#out + 1] = tryBool(o, "DED") and "1" or "0"
    out[#out + 1] = tostring(math.floor(tryNum(o, "Team Int", -1)))
    local okL, loc = tryCall(o, "K2_GetActorLocation")
    local x, y, z = 0, 0, 0
    if okL then x, y, z = vec(loc) end
    out[#out + 1] = f1(x); out[#out + 1] = f1(y); out[#out + 1] = f1(z)
    local okR, rot = tryCall(o, "K2_GetActorRotation")
    out[#out + 1] = f1(okR and rot and rot.Yaw or 0)
    local okV, vel = tryCall(o, "GetVelocity")
    local vx, vy, vz = 0, 0, 0
    if okV then vx, vy, vz = vec(vel) end
    out[#out + 1] = f1(vx); out[#out + 1] = f1(vy); out[#out + 1] = f1(vz)
    -- bones (in the configured order; missing bones are sent as the actor location with a 0 flag)
    out[#out + 1] = tostring(#cfg.bones)
    for _, b in ipairs(cfg.bones) do
        local have = 0
        local bx, by, bz = x, y, z
        if f.mesh ~= nil and f.bones ~= nil and f.bones[b] ~= nil then
            local ok, p = tryCall(f.mesh, "GetSocketLocation", f.bones[b])
            if ok and p then
                bx, by, bz = vec(p)
                have = 1
            end
        end
        out[#out + 1] = tostring(have)
        out[#out + 1] = f1(bx); out[#out + 1] = f1(by); out[#out + 1] = f1(bz)
    end
end

local function gameTime()
    local ok, t = pcall(function()
        local gs = UEHelpers.GetGameplayStatics()
        local world = UEHelpers.GetWorld()
        return gs:GetRealTimeSeconds(world)
    end)
    if ok and type(t) == "number" then return t end
    return os.clock()
end

-- ---------------------------------------------------------------------
--  commands from python (one per line in cmd.txt)
-- ---------------------------------------------------------------------
local function healFighter(o)
    trySet(o, "Health", 100.0)
    trySet(o, "Consciousness", 100.0)
    trySet(o, "Stamina", 100.0)
    trySet(o, "All Body Tonus", 100.0)
    trySet(o, "DED", false)
    for _, part in ipairs({ "Head Health", "Neck Health", "Body Health", "Arm_R Health", "Arm_L Health", "Leg_R Health", "Leg_L Health" }) do
        trySet(o, part, 100.0)
    end
end

local function killFighter(o)
    trySet(o, "Health", -1.0)
    tryCall(o, "Death")
end

local function spawnClassInFront(classPath, distanceCm, team)
    local pc = getPC()
    if not pc then return false end
    local pawn = tryGet(pc, "Pawn")
    if not isValid(pawn) then return false end
    local okA = pcall(function() LoadAsset(classPath) end)
    local cls = StaticFindObject(classPath)
    if not isValid(cls) then
        log("spawn: class not found " .. classPath)
        return false
    end
    local okL, loc = tryCall(pawn, "K2_GetActorLocation")
    local okR, rot = tryCall(pawn, "K2_GetActorRotation")
    if not okL then return false end
    local yaw = (okR and rot and rot.Yaw or 0) * math.pi / 180.0
    local x = loc.X + math.cos(yaw) * distanceCm
    local y = loc.Y + math.sin(yaw) * distanceCm
    local z = loc.Z + 20
    local faceYaw = ((okR and rot and rot.Yaw or 0) + 180.0)
    local world = pc:GetWorld()
    local ok, actor = pcall(function()
        return world:SpawnActor(cls, { X = x, Y = y, Z = z }, { Pitch = 0, Yaw = faceYaw, Roll = 0 })
    end)
    if ok and isValid(actor) then
        if team then trySet(actor, "Team Int", team) end
        cache.spawned[#cache.spawned + 1] = actor
        cache.forceRescan = true
        log("spawned " .. classPath)
        return true
    end
    log("spawn failed: " .. tostring(actor))
    return false
end

local function playerAndEnemies()
    local pc = getPC()
    local player = nil
    if pc then
        local pawn = tryGet(pc, "Pawn")
        if isValid(pawn) then player = pawn end
    end
    local playerTeam = player and tryNum(player, "Team Int", -1) or -1
    local enemies = {}
    for _, f in ipairs(cache.fighters) do
        if isValid(f.obj) and (player == nil or f.obj:GetAddress() ~= player:GetAddress()) then
            local t = tryNum(f.obj, "Team Int", -1)
            if t ~= playerTeam then enemies[#enemies + 1] = f end
        end
    end
    return player, enemies
end

local function handleCommand(line)
    local parts = {}
    for p in line:gmatch("[^|]+") do parts[#parts + 1] = p end
    local cmd = parts[1]
    if not cmd then return end
    local player, enemies = playerAndEnemies()
    if cmd == "heal" then
        if player then healFighter(player) end
    elseif cmd == "heal_enemies" then
        for _, f in ipairs(enemies) do healFighter(f.obj) end
    elseif cmd == "kill_enemies" then
        for _, f in ipairs(enemies) do killFighter(f.obj) end
    elseif cmd == "invuln" then
        if player then trySet(player, "Invulnerable", parts[2] == "1") end
    elseif cmd == "timedilation" then
        local v = tonumber(parts[2]) or 1.0
        local ok, ws = pcall(function() return UEHelpers.GetWorldSettings() end)
        if ok and isValid(ws) then trySet(ws, "TimeDilation", v) end
    elseif cmd == "spawn" then
        local dist = tonumber(parts[3]) or 300
        local team = tonumber(parts[4])
        if not team then
            local pt = player and tryNum(player, "Team Int", 1) or 1
            team = (pt == 2) and 3 or 2
        end
        spawnClassInFront(parts[2], dist, team)
    elseif cmd == "despawn_spawned" then
        for _, a in ipairs(cache.spawned) do
            if isValid(a) then tryCall(a, "K2_DestroyActor") end
        end
        cache.spawned = {}
        cache.forceRescan = true
    elseif cmd == "set" then
        if player and parts[2] then
            local v = tonumber(parts[3])
            if v == nil then v = (parts[3] == "true" or parts[3] == "1") end
            trySet(player, parts[2], v)
        end
    elseif cmd == "rescan" then
        cache.forceRescan = true
    elseif cmd == "debug" then
        cfg.debug = parts[2] == "1"
    end
end

local function pollCommands()
    local f = io.open(CMD_FILE, "r")
    if not f then return end
    local lines = {}
    for line in f:lines() do
        if line ~= "" then lines[#lines + 1] = line end
    end
    f:close()
    os.remove(CMD_FILE)
    for _, line in ipairs(lines) do
        local ok, err = pcall(handleCommand, line)
        if not ok then log("command failed: " .. tostring(err)) end
    end
end

-- ---------------------------------------------------------------------
--  per-frame tick (runs on the game thread)
-- ---------------------------------------------------------------------
local pending = false
local lastMapCheck = 0

local function tick()
    pending = false
    seq = seq + 1
    if cache.forceRescan or (nowMs() - cache.lastRescan) > cfg.rescan_ms then
        rescanFighters()
    end
    pollCommands()

    local player, enemies = playerAndEnemies()
    local px, py, pz = 0, 0, 0
    if player then
        local ok, loc = tryCall(player, "K2_GetActorLocation")
        if ok then px, py, pz = vec(loc) end
    end
    -- sort enemies by distance to the player
    for _, f in ipairs(enemies) do
        local ok, loc = tryCall(f.obj, "K2_GetActorLocation")
        local x, y, z = 0, 0, 0
        if ok then x, y, z = vec(loc) end
        f.dist = math.sqrt((x - px) ^ 2 + (y - py) ^ 2 + (z - pz) ^ 2)
    end
    table.sort(enemies, function(a, b) return (a.dist or 1e12) < (b.dist or 1e12) end)

    local out = { "HSAI", tostring(PROTO), tostring(seq), f2(gameTime()), cache.mapName }
    -- camera
    local camYaw, camPitch = 0, 0
    local pc = getPC()
    if pc then
        local rot = tryGet(pc, "ControlRotation")
        if rot then
            camYaw = rot.Yaw or 0
            camPitch = rot.Pitch or 0
        end
    end
    out[#out + 1] = f1(camYaw)
    out[#out + 1] = f1(camPitch)
    out[#out + 1] = tostring(#cache.fighters)
    -- player
    if player then
        out[#out + 1] = "P1"
        local pf = nil
        for _, f in ipairs(cache.fighters) do
            if isValid(f.obj) and f.obj:GetAddress() == player:GetAddress() then pf = f end
        end
        if pf == nil then pf = { obj = player } end
        fighterFields(pf, out)
    else
        out[#out + 1] = "P0"
    end
    -- enemies
    local n = math.min(#enemies, cfg.max_enemies)
    out[#out + 1] = "E" .. tostring(n)
    for i = 1, n do
        fighterFields(enemies[i], out)
    end
    out[#out + 1] = "END"
    emit(table.concat(out, "|"))
end

local function schedule()
    if pending then return false end
    pending = true
    ExecuteInGameThread(function()
        local ok, err = pcall(tick)
        if not ok then
            pending = false
            if cfg.debug then log("tick error: " .. tostring(err)) end
        end
    end)
    return false
end

-- ---------------------------------------------------------------------
--  start-up
-- ---------------------------------------------------------------------
writeSmallFile(STATUS_FILE, string.format("loaded=1\nproto=%d\ntime=%s\npipe=%s\nfile=%s\n",
    PROTO, os.date("%Y-%m-%d %H:%M:%S"), cfg.pipe, TELEMETRY_FILE))

RegisterHook("/Script/Engine.PlayerController:ClientRestart", function(self)
    cache.pc = nil
    cache.forceRescan = true
end)

LoopAsync(cfg.tick_ms, schedule)
log(string.format("telemetry mod loaded (proto %d), pipe=%s", PROTO, cfg.pipe))

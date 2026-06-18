#!/usr/bin/env node
/* Publish a planned JetArm pulse sequence through rosbridge.
 *
 * Default behavior is dry-run only. Use --execute to move the physical arm.
 */

const fs = require("fs");

function parseArgs(argv) {
  const args = {
    host: "192.168.12.89",
    port: 9090,
    topic: "/servo_controller",
    plan: "dry_run_plan.json",
    execute: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--execute") args.execute = true;
    else if (arg === "--host") args.host = argv[++i];
    else if (arg === "--port") args.port = Number(argv[++i]);
    else if (arg === "--topic") args.topic = argv[++i];
    else if (arg === "--plan") args.plan = argv[++i];
    else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadPlan(planPath) {
  const plan = JSON.parse(fs.readFileSync(planPath, "utf8"));
  const waypoints = plan.waypoints || [];
  for (const waypoint of waypoints) {
    if (!Array.isArray(waypoint.servos) || waypoint.servos.length === 0) {
      throw new Error(`Waypoint ${waypoint.name || "<unnamed>"} has no servo pulses`);
    }
    for (const [id, pulse] of waypoint.servos) {
      if (id < 1 || id > 10 || pulse < 0 || pulse > 1000) {
        throw new Error(`Unsafe servo pulse in ${waypoint.name}: id=${id}, pulse=${pulse}`);
      }
    }
  }
  return plan;
}

async function main() {
  const args = parseArgs(process.argv);
  const plan = loadPlan(args.plan);

  console.log(`Plan: ${plan.task || args.plan}`);
  for (const waypoint of plan.waypoints) {
    console.log(
      `${waypoint.name || "unnamed"} (${waypoint.duration || 1}s): ${JSON.stringify(waypoint.servos)}`
    );
  }

  if (!args.execute) {
    console.log("\nDRY RUN: no rosbridge messages were published. Add --execute to move the real arm.");
    return;
  }

  const url = `ws://${args.host}:${args.port}`;
  const ws = new WebSocket(url);

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`Timed out connecting to ${url}`)), 5000);
    ws.onopen = () => {
      clearTimeout(timeout);
      resolve();
    };
    ws.onerror = (error) => {
      clearTimeout(timeout);
      reject(error);
    };
  });

  ws.send(
    JSON.stringify({
      op: "advertise",
      topic: args.topic,
      type: "servo_controller_msgs/msg/ServosPosition",
    })
  );
  await sleep(200);

  for (const waypoint of plan.waypoints) {
    const duration = Number(waypoint.duration || 1.0);
    const msg = {
      duration,
      position_unit: "pulse",
      position: waypoint.servos.map(([id, pulse]) => ({ id, position: pulse })),
    };
    console.log(`EXECUTE ${waypoint.name || "unnamed"}: ${JSON.stringify(msg)}`);
    ws.send(JSON.stringify({ op: "publish", topic: args.topic, msg }));
    await sleep(Math.max(100, duration * 1000));
  }

  ws.send(JSON.stringify({ op: "unadvertise", topic: args.topic }));
  await sleep(100);
  ws.close();
  console.log("EXECUTION COMPLETE");
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});

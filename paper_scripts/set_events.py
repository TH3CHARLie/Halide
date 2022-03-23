import os
import subprocess


def get_next_event(comp_events, target_events):
    for e in comp_events:
        if e in target_events:
            return e
    return None

target_events = os.getenv('PAPI_TARGET_EVENTS')
if target_events is None:
    exit("PAPI_TARGET_EVENTS is not set.")

target_events = target_events.split(",")
if len(target_events) == 0:
    exit("PAPI_TARGET_EVENTS is not set.")

events = [target_events[0]]

while True:
    output = subprocess.run(["papi_event_chooser", "PRESET"] + events, stdout=subprocess.PIPE).stdout.decode("utf-8")
    comp_events = [line.split()[0] for line in output.split("\n") if "PAPI_" in line ]
    event = get_next_event(comp_events, target_events)
    if event is not None:
        events.append(event)
    else:
        break

remain_events = [event for event in target_events if event not in events]
print(",".join(events))
print(",".join(remain_events))
import json

with open(r'd:\DamonAI\ai\prompt_construction\data\raw_dialogues\Abigail.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect all unique context fields to understand what conditions we need
for d in data:
    ctrl = d.get('control', {})
    ctx = d.get('context', {})
    rel = ctx.get('relationship_status', '')
    route = ctrl.get('route', 'any')
    spoiler = ctrl.get('spoiler_level', 0)
    topic_tags = ctx.get('topic_tags', [])
    text = d['data']['text'][:60]
    id_ = d['id']
    
    print(f"{id_:45s} | rel={rel:12s} | route={route:24s} | spoiler={spoiler} | tags={topic_tags}")
    print(f"{'':45s} | \"{text}\"")
    print()

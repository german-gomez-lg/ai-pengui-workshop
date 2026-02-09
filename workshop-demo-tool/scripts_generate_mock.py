import json
import random
from pathlib import Path

random.seed(42)

clients = [
    "Ford","Disney","Nike","Coca-Cola","Samsung","Sony","Toyota","McDonald's",
    "Amazon","Google","Microsoft","Apple","Adidas","Pepsi","Walmart","Target",
    "Best Buy","Home Depot","IKEA","Starbucks","Uber","Airbnb","Lowe's",
    "Honda","Lexus","BMW","Mercedes","Hyundai","Kia","LG"
]

channels = ["CTV","Streaming","FAST","Digital Out-of-Home"]

def make_campaign(client, idx):
    base_impressions = random.randint(200_000, 1_200_000)
    base_clicks = int(base_impressions * random.uniform(0.002, 0.008))
    base_spend = round(base_impressions * random.uniform(0.010, 0.025), 2)
    base_conversions = int(base_clicks * random.uniform(0.04, 0.12))
    status = random.choices(["active","paused"],[0.75,0.25])[0]
    return {
        "campaign_id": f"CMP-{1000+idx}",
        "name": f"{client} {random.choice(['Awareness','Consideration','Promo','Launch','Seasonal'])}",
        "client": client,
        "channel": random.choice(channels),
        "geo": "US",
        "status": status,
        "base_metrics": {
            "impressions": base_impressions,
            "clicks": base_clicks,
            "spend": base_spend,
            "conversions": base_conversions,
        },
    }

campaigns = []
idx = 1
for client in clients:
    for _ in range(random.randint(1, 2)):
        campaigns.append(make_campaign(client, idx))
        idx += 1

out = {"campaigns": campaigns}
path = Path("/home/will/dev/test-pengui-deploy/workshop-demo-tool/data/mock_campaigns.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Wrote {len(campaigns)} campaigns")

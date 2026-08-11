# Expected Value Router (AdTech)

Routes traffic to advertisers while optimizing the expected value.

### High Level Flow
```mermaid
---
config:
  layout: elk
---
flowchart LR
    A["Candidate Generation<br/>(prob of advertiser for a given user)"] --> B["Prob of conversion<br/>given user and advertiser"]
    B --> C["EV = Prob(conversion)<br/>× book_value<br/>× commission_rate"]
    C --> D["Rank advertisers for each user<br/>based on EV"]
    D --> E["Match user with advertiser<br/>based on exploration vs exploitation"]
```


### TODO




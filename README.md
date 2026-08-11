# Expected Value Router (AdTech)

Routes traffic to advertisers while optimizing the expected value.

### High Level Flow
```mermaid
flowchart LR
    A["Candidate Generation<br/>probability of advertiser for a given user"] -->
    B["Probability of conversion<br/>given user and advertiser"]
    B --> C["Expected value (EV)<br/>probability of conversion × book value × commission rate"]
    C --> D["Rank advertisers for each user<br/>based on EV"]
    D --> E["Match user with advertiser<br/>using exploration vs. exploitation"]
```


### TODO




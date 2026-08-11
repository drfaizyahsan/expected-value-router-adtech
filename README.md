# Expected Value Router (AdTech)

Routes traffic to advertisers while optimizing the expected value.

### High Level Flow
```mermaid
flowchart LR
    A[Candidate Generation i.e. prob of advertiser for a given user] --> B[Prob of conversion given user and advertiser]
    B --> C[EV = Prob(conversion) &times; book_value &times; commission_rate]
    C --> D[rank advertisers for each user based on EV]
    D --> E[match user with advertiser based on exploration v/s exploitation]
```

### TODO




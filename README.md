# Trading Algorithm

- I am using genetic algorithms to find the best offspring that will bring in the maximum returns.
- I am training 2 populations in parallel, one for short only and the other for long only trades.
- Then, using PPO optimization I train an agent to allocate in the testing phase capital to both strategies
- The ideal allocation beats the market both in terms of return and risk adjusted
- My purpose is to get as close to the ideal allocation and deploy my trading bot live.
## EDTH Hackathon Berlin, Top 15 of all Teams

Building Simulation Environment to test different detection methods for low flying attack drones that cannot be detected well by radar due to occlusion or radar simply being to expensive while also incorporating the time to intercept. 

# How to Run: 

First initialize the uv venv environment. Then, you can do a full simulation via
```
uv run src/trial_runner.py
```
To customize it, you need to change the main function in trial_runner.py. For example you can integrate self generated json data which you can do at 

```
uv run -m http.server 5001 -d ./frontend
```

To generate scenarios and get more visualizations, you can use the gradio app with 

```
uv run gradio src/vis.py
```

----

# Further work 

Currently only supporting detection, integrate the interceptors. 

Make the statistical analysis robust, i.e. remove independence assumption (currently probability for a window is just the product over the timesteps) and replace with at least markovian stuff. 

Integrate better trajectories, i.e. have multiple attack patterns that can be simulated via some neat ODEs or PDEs (i.e. possibly Diffusion Equations, or Mean field games if wanting something fancy)



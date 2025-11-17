## Code for ETDH Hackathon Berlin

Building Simulation Environment to test different detection methods for low flying attack drones that cannot be detected well by radar due to occlusion or radar simply being to expensive while also incorporating the time to intercept. 

----

Currently only supporting detection simulation with statistical analysis. 
Further work is also simulating interception.


#### Setup:

To start a full run, first initialize the venv. For that do 
```
uv venv
```
then do 
```
.venv/bin/activate
```
and 
```
uv sync
```
# How to Run: 

Now, you can do a fully simulation run via 
```
uv run src/trial_runner.py
```

or to generate nice Json data you can do 

```
uv run -m http.server 5001 -d ./frontend
```

To generate nice grids and get nice visualizations, you can use the gradio app with 

```
uv run gradio vis.py
```
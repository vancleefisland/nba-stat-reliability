"""Interactive tool: enter a player's makes and attempts, get a shrunk estimate with a range.
Run:  streamlit run app.py   (after `python -m src.run_pipeline` has created outputs/priors.json)"""
import json
import streamlit as st
from src.shrinkage import BetaPrior

st.title("Is this number real?")
st.caption("Shrinks a small-sample stat toward the league average, in proportion to how little data we have.")

try:
    priors = json.load(open("outputs/priors.json"))
except FileNotFoundError:
    st.error("Run `python -m src.run_pipeline` first to create outputs/priors.json.")
    st.stop()

stat = st.selectbox("Stat", list(priors))
p = BetaPrior(priors[stat]["alpha"], priors[stat]["beta"])
made = st.number_input("Makes / successes", min_value=0, value=12, step=1)
att = st.number_input("Attempts", min_value=1, value=25, step=1)
level = st.slider("Range width", 0.5, 0.95, 0.8)

if made > att:
    st.warning("Makes cannot exceed attempts.")
else:
    est = float(p.posterior_mean(made, att))
    lo, hi = (float(v) for v in p.posterior_interval(made, att, level))
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw", f"{made / att:.1%}")
    c2.metric("Best estimate of true skill", f"{est:.1%}")
    c3.metric(f"{level:.0%} range", f"{lo:.1%} to {hi:.1%}")
    st.write(f"League average is **{p.mean:.1%}**. This stat has the weight of about "
             f"**{p.strength:.0f} league-average attempts** behind it, so with only {att} attempts "
             f"the estimate stays close to the league average.")

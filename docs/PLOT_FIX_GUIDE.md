# Plot fix

This patch fixes the Streamlit crash caused by the bar-chart helper.

The previous helper used Plotly Express with pandas Index objects. On Streamlit Cloud, that can raise a redacted ValueError inside Plotly Express build_dataframe.

The new helper uses plotly.graph_objects and converts values/labels to plain Python lists before plotting.

Replace `app/live_dashboard.py`, then reboot Streamlit Cloud.

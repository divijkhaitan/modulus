import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings

def create_interactive_forecast_comparison_html(
    array1,
    array2,
    base_time_array,
    variable_names,
    lead_time_interval_hours=6,
    output_html_file="forecast_comparison.html"
):
    """
    Creates a standalone HTML file with an interactive comparison of two 5D
    forecast datasets using Plotly.

    Allows selection via three separate dropdown menus:
    - Dropdown 1: Initialization Time
    - Dropdown 2: Variable
    - Dropdown 3: Lead Time

    Selecting an option in one dropdown resets the view based on that selection,
    assuming the other two dimensions are at their initial index (0).

    Parameters:
    -----------
    array1 : numpy.ndarray
        Groundtruth observation values
        (n_init_times, n_lead_times, n_variables, n_lat, n_lon).
    array2 : numpy.ndarray
        Predicted forecast values with the same shape as array1.
    base_time_array : list or numpy.ndarray
        List/array of starting times (datetime objects or parsable strings)
        corresponding to the first dimension (n_init_times).
    variable_names : list or numpy.ndarray
        List/array of strings for the variable names corresponding to the
        third dimension (n_variables).
    lead_time_interval_hours : int or float, optional
        Number of hours between steps in the lead time dimension (n_lead_times).
        Defaults to 1.
    output_html_file : str, optional
        Path to save the generated HTML file.
        Defaults to "forecast_comparison_3_dropdowns.html".

    Returns:
    --------
    str
        The generated HTML content as a string. Also saves the HTML to
        `output_html_file`.
    """
    if not isinstance(array1, np.ndarray) or array1.ndim != 5:
        raise ValueError("array1 must be a 5D numpy array (n_init_times, n_lead_times, n_variables, n_lat, n_lon)")
    if array1.shape != array2.shape:
        raise ValueError("array1 and array2 must have the same shape")
    if len(base_time_array) != array1.shape[0]:
        raise ValueError(f"Length of base_time_array ({len(base_time_array)}) must match the first dimension ({array1.shape[0]})")
    if len(variable_names) != array1.shape[2]:
        raise ValueError(f"Length of variable_names ({len(variable_names)}) must match the third dimension ({array1.shape[2]})")

    n_init_times, n_lead_times, n_variables, n_lat, n_lon = array1.shape

    base_time_labels = []
    base_time_datetimes = []
    print("Parsing base times...")
    for i, t in enumerate(base_time_array):
        dt_obj = None
        if isinstance(t, datetime):
            dt_obj = t
            label = t.strftime('%Y-%m-%d %H:%M')
            processed = True
        elif isinstance(t, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt_obj = datetime.strptime(t, fmt)
                    label = dt_obj.strftime('%Y-%m-%d %H:%M')
                    processed = True
                    break
                except ValueError:
                    continue
            if not processed:
                label = f"Parse Error: {t}"
        else:
                label = f"Unknown Type: {t}"

        base_time_labels.append(label)
        base_time_datetimes.append(dt_obj)

    print("Base time parsing complete.")

    memo = {}
    def get_state_data(init_idx, lead_idx, var_idx):
        state_key = (init_idx, lead_idx, var_idx)
        if state_key in memo:
            return memo[state_key]

        img1 = array1[init_idx, lead_idx, var_idx, :, :]
        img2 = array2[init_idx, lead_idx, var_idx, :, :]

        min_val = np.nanmin([np.nanmin(img1), np.nanmin(img2)])
        max_val = np.nanmax([np.nanmax(img1), np.nanmax(img2)])

        if min_val == max_val:
            min_val = min_val - 0.5
            max_val = max_val + 0.5

        current_lead_hours = lead_idx * lead_time_interval_hours
        variable_name = variable_names[var_idx] if 0 <= var_idx < len(variable_names) else f"Var Index {var_idx} Error"
        init_time_label = base_time_labels[init_idx] if 0 <= init_idx < len(base_time_labels) else f"Init Index {init_idx} Error"
        init_dt = base_time_datetimes[init_idx] if 0 <= init_idx < len(base_time_datetimes) else None

        valid_time_str = "Valid: ?"
        if init_dt:
            valid_dt = init_dt + timedelta(hours=current_lead_hours)
            valid_time_str = f"Valid: {valid_dt.strftime('%a %d %b %H:%M')}"
        elif "Error" in init_time_label:
                valid_time_str = "Valid: Init Time Error"

        title_text = f"{variable_name}<br>Init: {init_time_label} | Lead: {current_lead_hours:.1f}h | {valid_time_str}"

        result = {
            'z1': img1, 'z2': img2,
            'min_val': min_val, 'max_val': max_val, 'title': title_text,
            'current_lead_hours': current_lead_hours,
            'init_time_label': init_time_label, 'variable_name': variable_name,
        }
        memo[state_key] = result
        return result

    print("Creating initial figure...")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Groundtruth", "Predicted"),
        shared_yaxes=True
    )

    initial_state = get_state_data(init_idx=0, lead_idx=0, var_idx=0)

    fig.add_trace(
        go.Heatmap(
            z=initial_state['z1'], zmin=initial_state['min_val'], zmax=initial_state['max_val'],
            colorscale='Viridis', colorbar=dict(title="Value", thickness=15),
            hoverinfo='skip', name='Groundtruth', uid='trace_arr1'
        ), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=initial_state['z2'], zmin=initial_state['min_val'], zmax=initial_state['max_val'],
            colorscale='Viridis', showscale=False, hoverinfo='skip',
            name='Predicted', uid='trace_arr2'
        ), row=1, col=2
    )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x1", scaleratio=1)
    fig.update_layout(yaxis2=dict(scaleanchor="x2", scaleratio=1))

    print("Creating interactive dropdown controls...")

    # Dropdown 1: Initialization Time
    init_time_buttons = []
    for init_idx in range(n_init_times):
        # Action: Update plot to show this init time, resetting lead and var to 0
        state_data = get_state_data(init_idx, 0, 0)
        label = base_time_labels[init_idx] if 0 <= init_idx < len(base_time_labels) else f"Index {init_idx}"

        button = dict(
            method="update",
            args=[
                {'z': [state_data['z1'], state_data['z2']],
                 'zmin': [state_data['min_val'], state_data['min_val']],
                 'zmax': [state_data['max_val'], state_data['max_val']],
                 'traces': [0, 1]},
                {'title': state_data['title']}
            ],
            label=label
        )
        init_time_buttons.append(button)

    # Dropdown 2: Variable
    variable_buttons = []
    for var_idx in range(n_variables):
        # Action: Update plot to show this variable, resetting init and lead to 0
        state_data = get_state_data(0, 0, var_idx)
        label = variable_names[var_idx] if 0 <= var_idx < len(variable_names) else f"Index {var_idx}"

        button = dict(
            method="update",
            args=[
                 {'z': [state_data['z1'], state_data['z2']],
                  'zmin': [state_data['min_val'], state_data['min_val']],
                  'zmax': [state_data['max_val'], state_data['max_val']],
                  'traces': [0, 1]},
                 {'title': state_data['title']}
            ],
            label=label
        )
        variable_buttons.append(button)

    # Dropdown 3: Lead Time
    lead_time_buttons = []
    for lead_idx in range(n_lead_times):
        # Action: Update plot to show this lead time, resetting init and var to 0
        state_data = get_state_data(0, lead_idx, 0)
        label = f"{lead_idx * lead_time_interval_hours:.1f} hr"

        button = dict(
            method="update",
            args=[
                 {'z': [state_data['z1'], state_data['z2']],
                  'zmin': [state_data['min_val'], state_data['min_val']],
                  'zmax': [state_data['max_val'], state_data['max_val']],
                  'traces': [0, 1]},
                 {'title': state_data['title']}
            ],
            label=label
        )
        lead_time_buttons.append(button)


    print("Updating layout with controls...")
    fig.update_layout(
        title=initial_state['title'],
        coloraxis1=dict(
             colorscale='Viridis',
             colorbar=dict(title="Value", thickness=15, len=0.8, y=0.5, yanchor='middle')
        ),
        updatemenus=[
            dict(
                buttons=init_time_buttons,
                direction="down", pad={"r": 10, "t": 10}, showactive=True,
                x=0.02, xanchor="left",
                y=1.18, yanchor="top",
                active=0, name="Init Time"
            ),
            dict(
                buttons=variable_buttons,
                direction="down", pad={"r": 10, "t": 10}, showactive=True,
                x=0.35, xanchor="left", 
                y=1.18, yanchor="top",
                active=0, name="Variable"
            ),
            dict(
                buttons=lead_time_buttons,
                direction="down", pad={"r": 10, "t": 10}, showactive=True,
                x=0.68, xanchor="left",                 
                y=1.18, yanchor="top",
                active=0, name="Lead Time"
            )
        ],
        margin=dict(l=50, r=50, t=150, b=50)
    )

    print(f"Generating HTML file: {output_html_file}...")
    html_content = fig.to_html(
        full_html=True,
        include_plotlyjs='cdn'
    )

    with open(output_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Successfully saved interactive plot to {output_html_file}")
    
    return html_content

if __name__ == "__main__":
        n_inits = 3 
        n_leads = 12 
        n_vars = 4 
        n_lat, n_lon = 24, 48

        var_names = [f'Var_{chr(65+i)}' for i in range(n_vars)] 

        start_time = datetime(2025, 4, 17, 0, 0) 
        base_times = [(start_time + timedelta(hours=12 * i)).strftime('%Y-%m-%d %H:%M:%S')
                      for i in range(n_inits)]

        lead_interval_h = 6 

        shape = (n_inits, n_leads, n_vars, n_lat, n_lon)
        print(f"Generating dummy data with shape: {shape}")

        dummy_array1 = np.zeros(shape, dtype=np.float32)
        dummy_array2 = np.zeros(shape, dtype=np.float32)

        lon_grid, lat_grid = np.meshgrid(np.linspace(0, 2*np.pi, n_lon), np.linspace(-np.pi/2, np.pi/2, n_lat))
        base_pattern = np.cos(lat_grid) * np.sin(lon_grid) * 50 + 280

        for init_i in range(n_inits):
            for lead_i in range(n_leads):
                for var_i in range(n_vars):
                    variation1 = base_pattern + \
                                init_i * 5 + \
                                np.sin(lead_i * np.pi / n_leads) * 10 * (var_i + 1) + \
                                np.random.randn(n_lat, n_lon) * 2 
                    variation2 = base_pattern + \
                                init_i * 5.5 + \
                                np.cos(lead_i * np.pi / n_leads) * 11 * (var_i + 1) + \
                                np.random.randn(n_lat, n_lon) * 2.5

                    dummy_array1[init_i, lead_i, var_i, :, :] = variation1
                    dummy_array2[init_i, lead_i, var_i, :, :] = variation2

        print("Dummy data generated.")

        output_filename = "interactive_forecast_comparison_3_dropdowns.html"
        html_output = create_interactive_forecast_comparison_html(
            dummy_array1,
            dummy_array2,
            base_times,
            var_names,
            lead_time_interval_hours=lead_interval_h,
            output_html_file=output_filename
        )

        if html_output:
            print(f"\nTo view the plot, open the file '{output_filename}' in your web browser.")
        else:
            print("\nHTML generation failed.")

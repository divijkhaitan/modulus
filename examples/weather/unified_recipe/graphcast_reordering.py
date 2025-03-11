import numpy as np

ORIGINAL_ORDER_INPUTS_178 = [
    ('10m_u_component_of_wind', ('time', -1)),
    ('10m_u_component_of_wind', ('time', 0)),
    ('10m_v_component_of_wind', ('time', -1)),
    ('10m_v_component_of_wind', ('time', 0)),
    ('2m_temperature', ('time', -1)),
    ('2m_temperature', ('time', 0)),
    ('day_progress_cos', ('time', -1)),
    ('day_progress_cos', ('time', 0)),
    ('day_progress_sin', ('time', -1)),
    ('day_progress_sin', ('time', 0)),
    ('geopotential', ('time', -1), ('level', np.int64(50))),
    ('geopotential', ('time', -1), ('level', np.int64(100))),
    ('geopotential', ('time', -1), ('level', np.int64(150))),
    ('geopotential', ('time', -1), ('level', np.int64(200))),
    ('geopotential', ('time', -1), ('level', np.int64(250))),
    ('geopotential', ('time', -1), ('level', np.int64(300))),
    ('geopotential', ('time', -1), ('level', np.int64(400))),
    ('geopotential', ('time', -1), ('level', np.int64(500))),
    ('geopotential', ('time', -1), ('level', np.int64(600))),
    ('geopotential', ('time', -1), ('level', np.int64(700))),
    ('geopotential', ('time', -1), ('level', np.int64(850))),
    ('geopotential', ('time', -1), ('level', np.int64(925))),
    ('geopotential', ('time', -1), ('level', np.int64(1000))),
    ('geopotential', ('time', 0), ('level', np.int64(50))),
    ('geopotential', ('time', 0), ('level', np.int64(100))),
    ('geopotential', ('time', 0), ('level', np.int64(150))),
    ('geopotential', ('time', 0), ('level', np.int64(200))),
    ('geopotential', ('time', 0), ('level', np.int64(250))),
    ('geopotential', ('time', 0), ('level', np.int64(300))),
    ('geopotential', ('time', 0), ('level', np.int64(400))),
    ('geopotential', ('time', 0), ('level', np.int64(500))),
    ('geopotential', ('time', 0), ('level', np.int64(600))),
    ('geopotential', ('time', 0), ('level', np.int64(700))),
    ('geopotential', ('time', 0), ('level', np.int64(850))),
    ('geopotential', ('time', 0), ('level', np.int64(925))),
    ('geopotential', ('time', 0), ('level', np.int64(1000))),
    'geopotential_at_surface',
    'land_sea_mask',
    ('mean_sea_level_pressure', ('time', -1)),
    ('mean_sea_level_pressure', ('time', 0)),
    ('specific_humidity', ('time', -1), ('level', np.int64(50))),
    ('specific_humidity', ('time', -1), ('level', np.int64(100))),
    ('specific_humidity', ('time', -1), ('level', np.int64(150))),
    ('specific_humidity', ('time', -1), ('level', np.int64(200))),
    ('specific_humidity', ('time', -1), ('level', np.int64(250))),
    ('specific_humidity', ('time', -1), ('level', np.int64(300))),
    ('specific_humidity', ('time', -1), ('level', np.int64(400))),
    ('specific_humidity', ('time', -1), ('level', np.int64(500))),
    ('specific_humidity', ('time', -1), ('level', np.int64(600))),
    ('specific_humidity', ('time', -1), ('level', np.int64(700))),
    ('specific_humidity', ('time', -1), ('level', np.int64(850))),
    ('specific_humidity', ('time', -1), ('level', np.int64(925))),
    ('specific_humidity', ('time', -1), ('level', np.int64(1000))),
    ('specific_humidity', ('time', 0), ('level', np.int64(50))),
    ('specific_humidity', ('time', 0), ('level', np.int64(100))),
    ('specific_humidity', ('time', 0), ('level', np.int64(150))),
    ('specific_humidity', ('time', 0), ('level', np.int64(200))),
    ('specific_humidity', ('time', 0), ('level', np.int64(250))),
    ('specific_humidity', ('time', 0), ('level', np.int64(300))),
    ('specific_humidity', ('time', 0), ('level', np.int64(400))),
    ('specific_humidity', ('time', 0), ('level', np.int64(500))),
    ('specific_humidity', ('time', 0), ('level', np.int64(600))),
    ('specific_humidity', ('time', 0), ('level', np.int64(700))),
    ('specific_humidity', ('time', 0), ('level', np.int64(850))),
    ('specific_humidity', ('time', 0), ('level', np.int64(925))),
    ('specific_humidity', ('time', 0), ('level', np.int64(1000))),
    ('temperature', ('time', -1), ('level', np.int64(50))),
    ('temperature', ('time', -1), ('level', np.int64(100))),
    ('temperature', ('time', -1), ('level', np.int64(150))),
    ('temperature', ('time', -1), ('level', np.int64(200))),
    ('temperature', ('time', -1), ('level', np.int64(250))),
    ('temperature', ('time', -1), ('level', np.int64(300))),
    ('temperature', ('time', -1), ('level', np.int64(400))),
    ('temperature', ('time', -1), ('level', np.int64(500))),
    ('temperature', ('time', -1), ('level', np.int64(600))),
    ('temperature', ('time', -1), ('level', np.int64(700))),
    ('temperature', ('time', -1), ('level', np.int64(850))),
    ('temperature', ('time', -1), ('level', np.int64(925))),
    ('temperature', ('time', -1), ('level', np.int64(1000))),
    ('temperature', ('time', 0), ('level', np.int64(50))),
    ('temperature', ('time', 0), ('level', np.int64(100))),
    ('temperature', ('time', 0), ('level', np.int64(150))),
    ('temperature', ('time', 0), ('level', np.int64(200))),
    ('temperature', ('time', 0), ('level', np.int64(250))),
    ('temperature', ('time', 0), ('level', np.int64(300))),
    ('temperature', ('time', 0), ('level', np.int64(400))),
    ('temperature', ('time', 0), ('level', np.int64(500))),
    ('temperature', ('time', 0), ('level', np.int64(600))),
    ('temperature', ('time', 0), ('level', np.int64(700))),
    ('temperature', ('time', 0), ('level', np.int64(850))),
    ('temperature', ('time', 0), ('level', np.int64(925))),
    ('temperature', ('time', 0), ('level', np.int64(1000))),
    ('toa_incident_solar_radiation', ('time', -1)),
    ('toa_incident_solar_radiation', ('time', 0)),
    ('total_precipitation', ('time', -1)),
    ('total_precipitation', ('time', 0)),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(50))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(100))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(150))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(200))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(250))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(300))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(400))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(500))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(600))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(700))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(850))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(925))),
    ('u_component_of_wind', ('time', -1), ('level', np.int64(1000))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(50))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(100))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(150))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(200))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(250))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(300))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(400))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(500))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(600))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(700))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(850))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(925))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(1000))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(50))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(100))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(150))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(200))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(250))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(300))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(400))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(500))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(600))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(700))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(850))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(925))),
    ('v_component_of_wind', ('time', -1), ('level', np.int64(1000))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(50))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(100))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(150))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(200))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(250))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(300))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(400))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(500))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(600))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(700))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(850))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(925))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(1000))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(50))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(100))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(150))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(200))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(250))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(300))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(400))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(500))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(600))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(700))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(850))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(925))),
    ('vertical_velocity', ('time', -1), ('level', np.int64(1000))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(50))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(100))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(150))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(200))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(250))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(300))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(400))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(500))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(600))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(700))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(850))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(925))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(1000))),
    ('year_progress_cos', ('time', -1)),
    ('year_progress_cos', ('time', 0)),
    ('year_progress_sin', ('time', -1)),
    ('year_progress_sin', ('time', 0)),
]

ORIGINAL_ORDER_INPUTS_176 = [x for x in ORIGINAL_ORDER_INPUTS_178 if isinstance(x, str) or len(x) == 3 or x[0] != "total_precipitation"]

ORIGINAL_ORDER_OUTPUTS_83 = [
    ('10m_u_component_of_wind', ('time', 0)),
    ('10m_v_component_of_wind', ('time', 0)),
    ('2m_temperature', ('time', 0)),
    ('geopotential', ('time', 0), ('level', np.int64(50))),
    ('geopotential', ('time', 0), ('level', np.int64(100))),
    ('geopotential', ('time', 0), ('level', np.int64(150))),
    ('geopotential', ('time', 0), ('level', np.int64(200))),
    ('geopotential', ('time', 0), ('level', np.int64(250))),
    ('geopotential', ('time', 0), ('level', np.int64(300))),
    ('geopotential', ('time', 0), ('level', np.int64(400))),
    ('geopotential', ('time', 0), ('level', np.int64(500))),
    ('geopotential', ('time', 0), ('level', np.int64(600))),
    ('geopotential', ('time', 0), ('level', np.int64(700))),
    ('geopotential', ('time', 0), ('level', np.int64(850))),
    ('geopotential', ('time', 0), ('level', np.int64(925))),
    ('geopotential', ('time', 0), ('level', np.int64(1000))),
    ('mean_sea_level_pressure', ('time', 0)),
    ('specific_humidity', ('time', 0), ('level', np.int64(50))),
    ('specific_humidity', ('time', 0), ('level', np.int64(100))),
    ('specific_humidity', ('time', 0), ('level', np.int64(150))),
    ('specific_humidity', ('time', 0), ('level', np.int64(200))),
    ('specific_humidity', ('time', 0), ('level', np.int64(250))),
    ('specific_humidity', ('time', 0), ('level', np.int64(300))),
    ('specific_humidity', ('time', 0), ('level', np.int64(400))),
    ('specific_humidity', ('time', 0), ('level', np.int64(500))),
    ('specific_humidity', ('time', 0), ('level', np.int64(600))),
    ('specific_humidity', ('time', 0), ('level', np.int64(700))),
    ('specific_humidity', ('time', 0), ('level', np.int64(850))),
    ('specific_humidity', ('time', 0), ('level', np.int64(925))),
    ('specific_humidity', ('time', 0), ('level', np.int64(1000))),
    ('temperature', ('time', 0), ('level', np.int64(50))),
    ('temperature', ('time', 0), ('level', np.int64(100))),
    ('temperature', ('time', 0), ('level', np.int64(150))),
    ('temperature', ('time', 0), ('level', np.int64(200))),
    ('temperature', ('time', 0), ('level', np.int64(250))),
    ('temperature', ('time', 0), ('level', np.int64(300))),
    ('temperature', ('time', 0), ('level', np.int64(400))),
    ('temperature', ('time', 0), ('level', np.int64(500))),
    ('temperature', ('time', 0), ('level', np.int64(600))),
    ('temperature', ('time', 0), ('level', np.int64(700))),
    ('temperature', ('time', 0), ('level', np.int64(850))),
    ('temperature', ('time', 0), ('level', np.int64(925))),
    ('temperature', ('time', 0), ('level', np.int64(1000))),
    ('total_precipitation', ('time', 0)), 
    ('u_component_of_wind', ('time', 0), ('level', np.int64(50))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(100))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(150))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(200))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(250))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(300))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(400))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(500))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(600))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(700))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(850))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(925))),
    ('u_component_of_wind', ('time', 0), ('level', np.int64(1000))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(50))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(100))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(150))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(200))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(250))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(300))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(400))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(500))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(600))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(700))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(850))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(925))),
    ('v_component_of_wind', ('time', 0), ('level', np.int64(1000))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(50))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(100))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(150))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(200))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(250))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(300))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(400))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(500))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(600))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(700))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(850))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(925))),
    ('vertical_velocity', ('time', 0), ('level', np.int64(1000)))
]

FORCING_VARIABLES = ['day_progress_cos', 'day_progress_sin', 'toa_incident_solar_radiation', 'year_progress_cos', 'year_progress_sin',]

def sort_key(item):
    """Sort key function for features."""
    if isinstance(item, str):  # Timeless features
        return (0, item)  # Always prioritize timeless, then original order
    elif len(item) == 2:  # features at a given time
        name, time = item
        time_val = time[1]  # -1 or 0
        return (time_val + 1, name)  # Prioritize time, then name

    elif len(item) == 3:  # Features with time and level
        name, time, level = item
        time_val = time[1]
        level_val = int(level[1])  # Ensure level is treated as a number
        return (time_val + 2, name, level_val)

def reorder_features(original_order, inputs=True):
    """Reorders the features according to the specified rules."""
    time_0_features = [f for f in original_order if isinstance(f, tuple) and len(f) >= 2 and f[1][1] == 0 and f[0] not in FORCING_VARIABLES]
    time_0_features.sort(key=sort_key)
    if inputs:
        timeless_features = ['geopotential_at_surface', 'land_sea_mask']
        forcings_minus_1 = [f for f in original_order if isinstance(f, tuple) and len(f) >= 2 and f[1][1] == -1 and f[0] in FORCING_VARIABLES]
        forcings_0 = [f for f in original_order if isinstance(f, tuple) and len(f) >= 2 and f[1][1] == 0 and f[0] in FORCING_VARIABLES]
        time_minus_1_features = [f for f in original_order if isinstance(f, tuple) and len(f) >= 2 and f[1][1] == -1 and f[0] not in FORCING_VARIABLES]

        time_minus_1_features.sort(key=sort_key)
        forcings_minus_1.sort(key=sort_key)
        forcings_0.sort(key=sort_key)
        final_order = timeless_features + forcings_minus_1 + time_minus_1_features + forcings_0 + time_0_features

    else:
      final_order = time_0_features
    return final_order

def dataset_to_array(order, aux_dataset_items, dataset, aux_dataset=None):
    array = []
    for item in order:
        if isinstance(item, str):
            if item in aux_dataset_items:
                if aux_dataset is None:
                    raise ValueError("Need an Auxillary dataset to draw normalisation values from")
                array.append(aux_dataset[item].item())
            else:
                array.append(dataset[item].item())
        elif len(item) == 2:
            name, _ = item
            if name in aux_dataset_items:
                if aux_dataset is None:
                    raise ValueError("Need an Auxillary dataset to draw normalisation values from")
                array.append(aux_dataset[name].item())
            else:
                array.append(dataset[name].item())
        elif len(item) == 3:
            name, _, level = item
            if name in aux_dataset_items:
                if aux_dataset is None:
                    raise ValueError("Need an Auxillary dataset to draw normalisation values from")
                array.append(aux_dataset[name].sel(level=int(level[1])).item())
            else:
                array.append(dataset[name].sel(level=int(level[1])).item())
    return np.array(array)

sorted_time_zero_input_features = [f for f in ORIGINAL_ORDER_INPUTS_178 if isinstance(f, tuple) and len(f) >= 2 and f[1][1] == 0 and f[0] not in FORCING_VARIABLES]
sorted_time_zero_input_features.sort(key=sort_key)

sorted_time_zero_output_features = [f for f in ORIGINAL_ORDER_OUTPUTS_83 if isinstance(f, tuple) and len(f) >= 2 and f[1][1] == 0 and f[0] not in FORCING_VARIABLES]
sorted_time_zero_output_features.sort(key=sort_key)

reordered_features_178 = reorder_features(ORIGINAL_ORDER_INPUTS_178)
reordered_features_176 = reorder_features(ORIGINAL_ORDER_INPUTS_176)
reordered_features_list_outputs_83 = reorder_features(ORIGINAL_ORDER_OUTPUTS_83, inputs=False)

original_indices_inputs_178 = {feature: i for i, feature in enumerate(ORIGINAL_ORDER_INPUTS_178)}
original_indices_inputs_176 = {feature: i for i, feature in enumerate(ORIGINAL_ORDER_INPUTS_176)}
original_indices_outputs_83 = {feature: i for i, feature in enumerate(ORIGINAL_ORDER_OUTPUTS_83)}
reordered_indices_178 = {feature: i for i, feature in enumerate(reordered_features_178)}
reordered_indices_176 = {feature: i for i, feature in enumerate(reordered_features_176)}
reordered_indices_outputs_83 = {feature: i for i, feature in enumerate(reordered_features_list_outputs_83)}

# Fetch the location at which the subject of re-ordering has the required attribute
reorder_178_to_original_178 = [reordered_indices_178[feature] for feature in ORIGINAL_ORDER_INPUTS_178]
reorder_176_to_original_176 = [reordered_indices_176[feature] for feature in ORIGINAL_ORDER_INPUTS_176]

reorder_178_to_original_176 = [reordered_indices_178[feature] for feature in ORIGINAL_ORDER_INPUTS_176]
reorder_178_to_original_output = [reordered_indices_178[feature] for feature in ORIGINAL_ORDER_OUTPUTS_83]
reorder_output_to_original_output = [reordered_indices_outputs_83[feature] for feature in ORIGINAL_ORDER_OUTPUTS_83]

original_178_to_original_83 = [original_indices_inputs_178[feature] for feature in ORIGINAL_ORDER_OUTPUTS_83]
original_176_to_original_83 = [original_indices_inputs_176[feature] if feature in original_indices_inputs_176 else None for feature in ORIGINAL_ORDER_OUTPUTS_83]

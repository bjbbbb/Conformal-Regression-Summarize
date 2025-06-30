import numpy as np
import pandas as pd
import torch


def save_test_predictions(predictor, test_data_loader, device, run_number=None):
    """
    Save prediction results on test set:
      - x: first feature of one-dimensional data
      - y: true label
      - lower: lower bound of prediction interval
      - upper: upper bound of prediction interval
      - within_interval: 1 if prediction interval covers y, otherwise 0
    """
    predictions_list = []
    for batch_x, batch_y in test_data_loader:
        batch_x = batch_x.to(device)
        batch_x_np = batch_x.cpu().numpy()  # Convert Tensor to NumPy array
        with torch.no_grad():
            batch_predictions = predictor.predict(batch_x_np)  # Use NumPy array for prediction
        if isinstance(batch_predictions, list):
            batch_predictions = np.array(batch_predictions)
        batch_y_np = batch_y.cpu().numpy()
        for x_i, y_i, pred in zip(batch_x_np, batch_y_np, batch_predictions):
            # Compatible with prediction results as tuple, list or numpy array (take first two values as lower and upper bounds)
            if isinstance(pred, (tuple, list)) and len(pred) >= 2:
                lower = np.array([interval[0] for interval in pred])
                upper = np.array([interval[1] for interval in pred])
            elif isinstance(pred, np.ndarray) and pred.size >= 2:
                lower, upper = float(pred.flat[0]), float(pred.flat[1])
            else:
                lower, upper = -np.inf, np.inf

            indicator = 1 if (y_i >= lower and y_i <= upper) else 0
            predictions_list.append({
                'x': x_i[0],  # x is one-dimensional data, take the first feature
                'y': float(y_i),
                'lower': lower,
                'upper': upper,
                'within_interval': indicator
            })

    run_str = "" if run_number is None else f"_run{run_number}"
    file_name = f"chr_nn_test_predictions{run_str}.csv"
    pd.DataFrame(predictions_list).to_csv(file_name, index=False)
    print(f"Test set results saved to {file_name}")

    # Generate visualization
    visualize_predictions(file_name, run_number)


# Correct approach using direct alpha handling with NaN or mask
def visualize_predictions(predictions_file, run_number=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(12, 8))

    # Load data
    predictions_df = pd.read_csv(predictions_file)

    # Calculate average size metric (interval width)
    predictions_df['interval_size'] = predictions_df['upper'] - predictions_df['lower']
    avg_size = predictions_df['interval_size'].mean()

    # Set colors based on y values and whether they are within prediction intervals
    in_interval = predictions_df['y'].between(predictions_df['lower'], predictions_df['upper'])

    # Create color array
    colors = np.empty(len(predictions_df), dtype=object)

    # Set colors for different cases
    for i, row in predictions_df.iterrows():
        if row['y'] < 0.1 or row['y'] > 0.9:  # y values below 0.1 or above 0.9
            if in_interval[i]:
                colors[i] = (0.255, 0.412, 0.882)  # Within interval, blue
            else:
                colors[i] = 'green'  # Outside interval, green
        else:  # y values within 0.1-0.9 range
            if in_interval[i]:
                colors[i] = (0.1, 0.1, 0.1)  # Within interval, black
            else:
                colors[i] = (1.0, 0.1, 0.1)  # Outside interval, red

    # Draw scatter plot (70% transparency)
    plt.scatter(predictions_df['x'], predictions_df['y'], facecolors='none',
                edgecolors=colors, s=50, alpha=0.7, label='Data Points')

    # Prepare grid
    predictions_df = predictions_df.sort_values('x')
    x_min, x_max = predictions_df['x'].min(), predictions_df['x'].max()
    y_min = min(predictions_df['lower'].min(), predictions_df['y'].min())
    y_max = max(predictions_df['upper'].max(), predictions_df['y'].max())
    grid_size = 1000
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    density_grid = np.zeros((grid_size, grid_size))

    x_width_factor = 0.01
    for _, row in predictions_df.iterrows():
        x_val = row['x']
        lower = row['lower']
        upper = row['upper']
        x_range = np.abs(x_grid - x_val) < (x_grid[1] - x_grid[0]) * x_width_factor * grid_size / (x_max - x_min)
        x_indices = np.where(x_range)[0]
        y_lower_idx = np.abs(y_grid - lower).argmin()
        y_upper_idx = np.abs(y_grid - upper).argmin()
        for x_idx in x_indices:
            density_grid[y_lower_idx:y_upper_idx + 1, x_idx] += 1

    # Create richer colormap (with more colors)
    colors = [
        (0.0, 0.6, 0.9),  # Deep sea blue
        (0.2, 0.7, 0.9),  # Sky blue
        (0.3, 0.8, 0.7),  # Cyan green
        (0.5, 0.9, 0.3),  # Yellow green
        (0.8, 0.9, 0.2),  # Cyan yellow
        (0.957, 0.7, 0.235),  # Bright orange
        (0.957, 0.569, 0.235),  # Coral orange
        (0.9, 0.4, 0.3),  # Orange red
        (0.8, 0.3, 0.5),  # Rose red
        (0.6, 0.349, 0.718),  # Deep violet
        (0.4, 0.3, 0.8),  # Light purple
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Draw density plot (without normalization, using raw counts)
    mask = density_grid > 0
    plt.imshow(np.where(mask, density_grid, np.nan), extent=[x_min, x_max, y_min, y_max],
               origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=np.nanmax(density_grid))

    # Add colorbar (showing raw counts)
    cbar = plt.colorbar()
    # cbar.set_label('Overlap Count', fontsize=32)  # Change label to count

    # Draw light gray prediction lines
    for _, row in predictions_df.iterrows():
        plt.plot([row['x'], row['x']], [row['lower'], row['upper']], 'k-', alpha=0.1, linewidth=0.5)

    # Add yellow-green reference lines at y=0.1 and y=0.9
    plt.axhline(y=0.1, color='yellowgreen', linestyle='-', linewidth=3, alpha=0.8)
    plt.axhline(y=0.9, color='yellowgreen', linestyle='-', linewidth=3, alpha=0.8)

    # Set labels and title
    plt.xlabel('X', fontsize=32)
    plt.ylabel('Y', fontsize=32)
    run_str = "" if run_number is None else f" (Run {run_number})"

    # Display average size metric in title
    plt.title(f'CHR - Avg Interval Size: {avg_size:.4f}', fontsize=32)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # Save as PDF format
    run_suffix = "" if run_number is None else f"_run{run_number}"
    plt.savefig(f"chr_synthetic.pdf", dpi=300, bbox_inches='tight', format='pdf')
    print(f"chr(rf){run_suffix}.pdf")
    plt.close()
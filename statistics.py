import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

def read_data(file_path):
    """
    Read data from a CSV file and return a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame or None: Loaded data, or None if there's an error.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Ensure it is a valid CSV file.")
        return None

def analyze_data(data, indicators_to_plot, selected_country):
    """
    Analyze data for the specific indicators and country.

    Parameters:
    - data (pd.DataFrame): Loaded data.
    - indicators_to_plot (list): List of indicators to plot.
    - selected_country (str): Selected country.

    Returns:
    - None (prints analysis results and plots the chart).
    """
    # Convert years to integers
    years = data.columns[2:].astype(int)

    # Filter data
    selected_data = data[(data['Country Name'] == selected_country) & (data['Indicator Name'].isin(indicators_to_plot))]

    # Use DataFrame methods to select values for each indicator
    values1 = selected_data.loc[selected_data['Indicator Name'] == indicators_to_plot[0]].iloc[0, 2:].values.astype(float)
    values2 = selected_data.loc[selected_data['Indicator Name'] == indicators_to_plot[1]].iloc[0, 2:].values.astype(float)

    # Display descriptive statistics for the values
    print("Descriptive Statistics for Urban Population:")
    print(pd.Series(values1).describe())

    print("\nDescriptive Statistics for CO2 Emissions (kt):")
    print(pd.Series(values2).describe())

    # Calculate and display correlation
    correlation_coefficient, _ = pearsonr(values1, values2)
    print(f"\nCorrelation between Urban Population and CO2 Emissions: {correlation_coefficient:.2f}")

    # Perform linear regression
    regression_result = linregress(values1, values2)
    print("\nLinear Regression:")
    print(f"Slope: {regression_result.slope:.2f}")
    print(f"Intercept: {regression_result.intercept:.2f}")
    print(f"R-squared: {regression_result.rvalue**2:.2f}")

    # Transpose selected data for visualization
    transposed_data = selected_data.iloc[:, 2:].transpose()

    # Plot the clustered column chart
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    bar_positions1 = [pos - bar_width/2 for pos in range(len(years))]
    bar_positions2 = [pos + bar_width/2 for pos in range(len(years))]

    bar1 = plt.bar(bar_positions1, values1, width=bar_width, label=indicators_to_plot[0], color='Blue')
    bar2 = plt.bar(bar_positions2, values2, width=bar_width, label=indicators_to_plot[1], color='Orange')

    plt.title(f'{selected_country} - Urban Population and CO2 Emissions (kt) (2000-2020)')
    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.xticks(range(len(years)), years)  # Set the x-axis ticks to represent years
    plt.legend()
    plt.grid(axis='y')
    plt.show()

file_path = r'/content/Urban_Population.csv'
loaded_data = read_data(file_path)

if loaded_data is not None:
    print("Data loaded successfully:")
    print(loaded_data.head())

    indicators_to_plot = ['Urban population', 'CO2 emissions (kt)']
    selected_country = 'Switzerland'
    analyze_data(loaded_data, indicators_to_plot, selected_country)




import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    Read data from a CSV file and return a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame or None: Loaded data, or None if there's an error.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Ensure it is a valid CSV file.")
        return None

def analyze_data(data, selected_countries, selected_indicators):
    """
    Analyze data for selected indicators and countries.

    Parameters:
    - data (pd.DataFrame): Loaded data.
    - selected_countries (list): List of selected countries.
    - selected_indicators (list): List of selected indicators.

    Returns:
    - None (prints analysis results and plots the chart).
    """
    # Filter data for selected countries and indicators
    selected_data = data[(data['Country Name'].isin(selected_countries)) & (data['Indicator Name'].isin(selected_indicators))]

    # Set up bar positions and width
    bar_positions = range(len(selected_countries))
    bar_width = 0.35

    # Create subplots
    fig, ax = plt.subplots()

    # Plot bars for each indicator and country
    for i, indicator in enumerate(selected_indicators):
        for j, country in enumerate(selected_countries):
            subset_data = selected_data[(selected_data['Country Name'] == country) & (selected_data['Indicator Name'] == indicator)]
            ax.bar(bar_positions[j] + i * bar_width, subset_data.iloc[0, 2:], width=bar_width, label=f'{country} - {indicator}')

    # Set labels and title
    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(selected_countries)
    ax.set_xlabel('Country')
    ax.set_ylabel('Indicator Value')
    ax.set_title('Bar Chart for Selected Indicators and Countries')

    # Add legend
    ax.legend()

    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    print(selected_data.iloc[:, 2:].describe().transpose())

    # Transpose the data
    transposed_data = selected_data.transpose()

    # Display transposed data
    print("\nTransposed Data:")
    print(transposed_data)

    # Show the plot
    plt.show()

# Example usage:
file_path = r'/content/Urban_Population.csv'
loaded_data = read_data(file_path)

if loaded_data is not None:
    print("Data loaded successfully:")
    print(loaded_data.head())

    # Example analysis
    selected_countries = ['Australia', 'United Arab Emirates', 'Brazil', 'Canada', 'Switzerland', 'Germany', 'India', 'Japan', 'United States']
    selected_indicators = ['CO2 emissions (kt)']
    analyze_data(loaded_data, selected_countries, selected_indicators)



# Select countries and indicators of interest
selected_countries = ['Australia', 'United Arab Emirates', 'Brazil', 'Canada', 'Switzerland', 'Germany', 'India', 'Japan', 'United States']
selected_indicators = ['CO2 emissions (kt)']

# Filter data for selected countries and indicators
selected_data = data[(data['Country Name'].isin(selected_countries)) & (data['Indicator Name'].isin(selected_indicators))]

# Set up bar positions and width
bar_positions = range(len(selected_countries))
bar_width = 0.35

# Create subplots
fig, ax = plt.subplots()

# Plot bars for each indicator and country
for i, indicator in enumerate(selected_indicators):
    for j, country in enumerate(selected_countries):
        subset_data = selected_data[(selected_data['Country Name'] == country) & (selected_data['Indicator Name'] == indicator)]
        ax.bar(bar_positions[j] + i * bar_width, subset_data.iloc[0, 2:], width=bar_width, label=f'{country} - {indicator}')

# Set labels and title
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
ax.set_xticklabels(selected_countries)
ax.set_xlabel('Country')
ax.set_ylabel('Indicator Value')
ax.set_title('Bar Chart for Selected Indicators and Countries')

# Add legend
ax.legend()

# Show the plot
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    Read data from a CSV file and return a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame or None: Loaded data, or None if there's an error.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Ensure it is a valid CSV file.")
        return None

def analyze_data(selected_data, selected_indicator, selected_countries):
    """
    Analyze data for the selected indicator and countries.

    Parameters:
    - selected_data (pd.DataFrame): Selected data for analysis.
    - selected_indicator (str): Selected indicator.
    - selected_countries (list): List of selected countries.

    Returns:
    - None (prints analysis results and plots the chart).
    """
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    print(selected_data.iloc[:, 2:].describe().transpose())

    # Transpose the data
    transposed_data = selected_data.transpose()

    # Display transposed data
    print("\nTransposed Data:")
    print(transposed_data)

    # Plot time trends
    plt.figure(figsize=(16, 10))
    for country in selected_countries:
        country_data = selected_data[selected_data['Country Name'] == country]

        # Check if there is data for the country before plotting
        if not country_data.empty:
            plt.plot(country_data.columns[2:], country_data.iloc[0, 2:], label=country)


    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'Time Trends for {selected_indicator} Across Countries')
    plt.legend()

    # The y-axis limits are determined automatically

    plt.show()

file_path = r'/content/Urban_Population.csv'
loaded_data = read_data(file_path)

if loaded_data is not None:
    print("Data loaded successfully:")
    print(loaded_data.head())

 # Select specific indicator and countries for analysis
    selected_indicator = 'Urban population'
    selected_countries = ['Australia', 'United Arab Emirates', 'Brazil', 'Canada', 'Switzerland', 'Germany', 'India', 'Japan', 'United States']
    selected_data = loaded_data[(loaded_data['Indicator Name'] == selected_indicator) & (loaded_data['Country Name'].isin(selected_countries))]
    analyze_data(selected_data, selected_indicator, selected_countries)

   

import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    Read data from a CSV file and return a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame or None: Loaded data, or None if there's an error.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Ensure it is a valid CSV file.")
        return None

def analyze_data(selected_data, selected_indicator, selected_countries):
    """
    Analyze data for the selected indicator and countries.

    Parameters:
    - selected_data (pd.DataFrame): Selected data for analysis.
    - selected_indicator (str): Selected indicator.
    - selected_countries (list): List of selected countries.

    Returns:
    - None (prints analysis results and plots the chart).
    """
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    print(selected_data.iloc[:, 2:].describe().transpose())

    # Transpose the data
    transposed_data = selected_data.transpose()

    # Display transposed data
    print("\nTransposed Data:")
    print(transposed_data)

    # Scatter plot
    plt.figure(figsize=(12, 8))
    for country in selected_countries:
        country_data = selected_data[selected_data['Country Name'] == country]

        # Check if there is data for the country before plotting
        if not country_data.empty:
            plt.scatter(country_data.columns[2:], country_data.iloc[0, 2:], label=country)

    plt.xlabel('Year')
    plt.ylabel(f'{selected_indicator} Value')
    plt.title(f'Scatter Plot for {selected_indicator} Across Countries')
    plt.legend()
    plt.grid(True)
    plt.show()

file_path = r'/content/Urban_Population.csv'
loaded_data = read_data(file_path)

if loaded_data is not None:
    print("Data loaded successfully:")
    print(loaded_data.head())

  # Select specific indicator and countries for analysis
    selected_indicator = 'CO2 emissions (kt)'
    selected_countries = ['Australia', 'United Arab Emirates', 'Brazil', 'Canada', 'Switzerland', 'Germany', 'India', 'Japan', 'United States']
    selected_data = loaded_data[(loaded_data['Indicator Name'] == selected_indicator) & (loaded_data['Country Name'].isin(selected_countries))]
    analyze_data(selected_data, selected_indicator, selected_countries)




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    Read data from a CSV file and return a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame or None: Loaded data, or None if there's an error.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'. Ensure it is a valid CSV file.")
        return None

def analyze_data(selected_data, selected_country, selected_indicators):
    """
    Analyze data for correlation matrix.

    Parameters:
    - selected_data (pd.DataFrame): Selected data for analysis.
    - selected_country (str): Selected country.
    - selected_indicators (list): List of selected indicators.

    Returns:
    - None (prints analysis results and plots the heatmap).
    """
    # Display descriptive statistics
    print("\nDescriptive Statistics:")
    print(selected_data.iloc[:, 2:].describe().transpose())

    # Transpose the data
    transposed_data = selected_data.transpose()

    # Display transposed data
    print("\nTransposed Data:")
    print(transposed_data)

    # Create a correlation matrix for selected indicators
    pivot_data = selected_data.melt(id_vars=['Country Name', 'Indicator Name'], value_vars=years, var_name='Year', value_name='Value')
    pivot_data['Year'] = pivot_data['Year'].astype(int)  # Convert 'Year' to integer type
    correlation_matrix = pivot_data.pivot_table(index='Year', columns='Indicator Name', values='Value').corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap using seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Set the title
    plt.title(f'Correlation Matrix for {selected_country} - Selected Indicators')

    # Show the plot
    plt.show()

file_path = '/content/Urban_Population.csv'
loaded_data = read_data(file_path)

if loaded_data is not None:
    print("Data loaded successfully:")
    print(loaded_data.head())

    # Select specific indicator and countries for analysis
    selected_country = 'China'
    selected_indicators = ['Urban population', 'CO2 emissions (kt)']
    selected_data = loaded_data[(loaded_data['Country Name'] == selected_country) & (loaded_data['Indicator Name'].isin(selected_indicators))]
    analyze_data(selected_data, selected_country, selected_indicators)




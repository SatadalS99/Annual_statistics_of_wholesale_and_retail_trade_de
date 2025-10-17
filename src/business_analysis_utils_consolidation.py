"""
Consolidated Business Analysis Utility Functions
Contains the most important analysis functions from 5 modules:
1. Profitability Trends Analysis
2. Sectoral Capital Dynamics
3. Labor Utilization Efficiency
4. Firm Size and Productivity
5. Temporal Growth Patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

def setup_plotting_style():
    """Set up consistent matplotlib and seaborn styles"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")


# ============================================================================
# 1. PROFITABILITY TRENDS ANALYSIS
# ============================================================================

def analyze_profitability_trends(df, metric='Gross yield ratio'):
    """
    Analyze and plot overall profitability trends over time
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Year, Gross yield ratio
    metric : str
        Column name for profitability metric
        
    Returns:
    --------
    yearly_stats : pd.DataFrame
        Yearly profitability statistics
    """
    print("\n" + "="*60)
    print("1. PROFITABILITY TRENDS ANALYSIS")
    print("="*60)
    
    # Clean data
    df_clean = df.dropna(subset=[metric, 'Year']).copy()
    df_clean = df_clean[
        (df_clean[metric] >= 0) & 
        (df_clean[metric] <= df_clean[metric].quantile(0.99))
    ]
    
    # Calculate yearly statistics
    yearly_stats = df_clean.groupby('Year')[metric].agg([
        'mean', 'median', 'std', 'count',
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75))
    ])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(yearly_stats.index, yearly_stats['mean'], 
             marker='o', linewidth=3, label='Mean', color='#2E86AB', markersize=8)
    
    plt.plot(yearly_stats.index, yearly_stats['median'], 
             marker='s', linewidth=2.5, label='Median', color='#A23B72', 
             linestyle='--', markersize=7)
    
    plt.fill_between(yearly_stats.index, 
                     yearly_stats['mean'] - yearly_stats['std'],
                     yearly_stats['mean'] + yearly_stats['std'],
                     alpha=0.2, color='#2E86AB', label='±1 Std Dev')
    
    plt.title('Overall Profitability Trends Over Time', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Gross Yield Ratio (%)', fontsize=12)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add sample size annotations
    for year in yearly_stats.index[::3]:
        count = yearly_stats.loc[year, 'count']
        plt.text(year, yearly_stats.loc[year, 'mean'], 
                 f'n={int(count)}', fontsize=8, ha='center', va='bottom', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nOverall Statistics:")
    print(f"  Mean:   {df_clean[metric].mean():.2f}%")
    print(f"  Median: {df_clean[metric].median():.2f}%")
    print(f"  Std:    {df_clean[metric].std():.2f}%")
    print(f"  Years:  {df_clean['Year'].min()} - {df_clean['Year'].max()}")
    print(f"  Total observations: {len(df_clean):,}")
    
    return yearly_stats


# ============================================================================
# 2. SECTORAL CAPITAL DYNAMICS
# ============================================================================

def analyze_capital_formation_trends(df):
    """
    Analyze temporal evolution of investment intensity
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Year, Gross capital formation, Turnover
        
    Returns:
    --------
    yearly_investment : pd.DataFrame
        Yearly investment statistics
    """
    print("\n" + "="*60)
    print("2. CAPITAL FORMATION TRENDS ANALYSIS")
    print("="*60)
    
    # Clean and prepare data
    df_clean = df.dropna(subset=['Gross capital formation', 'Turnover', 'Year']).copy()
    df_clean = df_clean[(df_clean['Turnover'] > 0) & 
                        (df_clean['Gross capital formation'] >= 0)]
    
    # Calculate Capital Formation Ratio
    df_clean['Capital_Formation_Ratio'] = (
        df_clean['Gross capital formation'] / df_clean['Turnover']
    )
    
    # Remove outliers
    cfr_99 = df_clean['Capital_Formation_Ratio'].quantile(0.99)
    df_clean = df_clean[df_clean['Capital_Formation_Ratio'] <= cfr_99]
    
    # Calculate yearly statistics
    yearly_investment = df_clean.groupby('Year').agg({
        'Capital_Formation_Ratio': ['mean', 'median', 'std', 'count'],
        'Gross capital formation': 'sum',
        'Turnover': 'sum'
    })
    
    yearly_investment.columns = ['_'.join(col).strip() for col in yearly_investment.columns]
    yearly_investment['Market_CFR'] = (
        yearly_investment['Gross capital formation_sum'] / 
        yearly_investment['Turnover_sum']
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    mean_cfr = yearly_investment['Capital_Formation_Ratio_mean']
    std_dev = yearly_investment['Capital_Formation_Ratio_std']
    
    ax.plot(yearly_investment.index, mean_cfr, 
            marker='o', linewidth=2.5, label='Average CFR', 
            color='#2E86AB', markersize=8, markeredgecolor='white', 
            markeredgewidth=1.5, zorder=3)
    
    ax.fill_between(yearly_investment.index,
                    mean_cfr - std_dev,
                    mean_cfr + std_dev,
                    alpha=0.25, color='#2E86AB', label='±1 Std Dev')
    
    # Add trend line
    years_numeric = yearly_investment.index.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        years_numeric, mean_cfr
    )
    trend_line = slope * years_numeric + intercept
    ax.plot(yearly_investment.index, trend_line, 
            color='#F18F01', linestyle=':', linewidth=2, 
            label=f'Trend (R²={r_value**2:.3f})', alpha=0.8, zorder=2)
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Capital Formation Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Investment Intensity Over Time', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAverage Capital Formation Ratio: {mean_cfr.mean():.4f}")
    print(f"Trend: {'Increasing' if slope > 0 else 'Decreasing'} ({slope:.6f} per year)")
    print(f"R² of trend: {r_value**2:.3f}")
    
    return yearly_investment


# ============================================================================
# 3. LABOR UTILIZATION EFFICIENCY
# ============================================================================

def analyze_labor_productivity_by_size(df):
    """
    Analyze labor productivity across different employee size classes
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Employee_Size_Class, Turnover per person employed
        
    Returns:
    --------
    size_stats : pd.DataFrame
        Statistics by size class
    """
    print("\n" + "="*60)
    print("3. LABOR PRODUCTIVITY BY SIZE CLASS")
    print("="*60)
    
    # Clean data
    df_clean = df.dropna(subset=['Employee_Size_Class', 'Turnover per person employed']).copy()
    df_clean = df_clean[df_clean['Turnover per person employed'] > 0]
    
    # Remove outliers
    turnover_99 = df_clean['Turnover per person employed'].quantile(0.995)
    df_clean = df_clean[df_clean['Turnover per person employed'] <= turnover_99]
    
    # Define size order
    size_order = [
        '1 to 2 persons employed',
        '3 to 5 persons employed', 
        '6 to 19 persons employed',
        '20 or more persons employed'
    ]
    
    # Filter to relevant size classes
    df_clean = df_clean[df_clean['Employee_Size_Class'].isin(size_order)]
    
    # Calculate statistics
    size_stats = df_clean.groupby('Employee_Size_Class')['Turnover per person employed'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reindex(size_order)
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.boxplot(data=df_clean, x='Employee_Size_Class', y='Turnover per person employed',
                order=size_order, palette='Set2', ax=ax)
    
    ax.set_title('Labor Productivity by Employee Size Class', fontsize=14, fontweight='bold')
    ax.set_xlabel('Employee Size Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Turnover per Person Employed (€)', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nProductivity Statistics by Size Class:")
    for size in size_order:
        if size in size_stats.index:
            stats_row = size_stats.loc[size]
            print(f"\n{size}:")
            print(f"  Mean:   €{stats_row['mean']:,.0f}")
            print(f"  Median: €{stats_row['median']:,.0f}")
            print(f"  Count:  {int(stats_row['count']):,}")
    
    return size_stats


# ============================================================================
# 4. FIRM SIZE AND PRODUCTIVITY
# ============================================================================

def analyze_size_productivity_relationship(df):
    """
    Analyze relationship between firm size and productivity with regression
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Employee_Size_Class, Turnover per enterprise
        
    Returns:
    --------
    correlation : float
        Pearson correlation coefficient
    """
    print("\n" + "="*60)
    print("4. FIRM SIZE VS PRODUCTIVITY RELATIONSHIP")
    print("="*60)
    
    # Clean data
    df_clean = df.dropna(subset=['Employee_Size_Class', 'Turnover per enterprise']).copy()
    
    # Define size encoding
    size_order = [
        "1 to 2 persons employed",
        "3 to 5 persons employed",
        "6 to 19 persons employed",
        "20 or more persons employed"
    ]
    size_encoding = {size: i for i, size in enumerate(size_order)}
    
    df_clean = df_clean[df_clean['Employee_Size_Class'].isin(size_order)]
    df_clean['Size_Encoded'] = df_clean['Employee_Size_Class'].map(size_encoding)
    
    # Remove outliers
    turnover_99 = df_clean['Turnover per enterprise'].quantile(0.995)
    df_clean = df_clean[df_clean['Turnover per enterprise'] <= turnover_99]
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample for visualization
    sample_df = df_clean.sample(n=min(5000, len(df_clean)), random_state=42)
    
    ax.scatter(sample_df['Size_Encoded'], sample_df['Turnover per enterprise'],
               alpha=0.5, s=20, color='coral')
    
    # Add regression line
    z = np.polyfit(df_clean['Size_Encoded'], df_clean['Turnover per enterprise'], 1)
    p = np.poly1d(z)
    ax.plot([0, 1, 2, 3], p([0, 1, 2, 3]), "r--", linewidth=2, label='Regression Line')
    
    ax.set_title('Firm Size vs Productivity Relationship', fontsize=14, fontweight='bold')
    ax.set_xlabel('Size Class', fontsize=12)
    ax.set_ylabel('Turnover per Enterprise (€)', fontsize=12)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['1-2', '3-5', '6-19', '20+'], rotation=0)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = stats.pearsonr(df_clean['Size_Encoded'], 
                                  df_clean['Turnover per enterprise'])[0]
    
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPearson Correlation: {correlation:.3f}")
    print(f"Regression equation: y = {z[0]:.2f}x + {z[1]:.2f}")
    print(f"Interpretation: {'Positive' if correlation > 0 else 'Negative'} relationship")
    
    return correlation


# ============================================================================
# 5. TEMPORAL GROWTH PATTERNS
# ============================================================================

def analyze_enterprise_growth_patterns(df, year_threshold=2005):
    """
    Analyze temporal evolution of enterprises and local units
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Year, Enterprises, Local units
    year_threshold : int
        Minimum year to include in analysis
        
    Returns:
    --------
    yearly_totals : pd.DataFrame
        Yearly totals and growth rates
    """
    print("\n" + "="*60)
    print("5. ENTERPRISE GROWTH PATTERNS ANALYSIS")
    print("="*60)
    
    # Filter and clean data
    df_clean = df[df['Year'] >= year_threshold].copy()
    df_clean = df_clean.dropna(subset=['Enterprises', 'Local units'])
    
    # Calculate yearly totals
    yearly_totals = df_clean.groupby('Year')[['Enterprises', 'Local units']].sum().reset_index()
    
    # Calculate growth rates
    yearly_totals['Enterprises_growth'] = yearly_totals['Enterprises'].pct_change() * 100
    yearly_totals['Local_units_growth'] = yearly_totals['Local units'].pct_change() * 100
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Absolute values
    ax1.plot(yearly_totals['Year'], yearly_totals['Enterprises'], 
             marker='o', linewidth=2, label='Total Enterprises', color='#2E86AB')
    ax1.plot(yearly_totals['Year'], yearly_totals['Local units'], 
             marker='s', linewidth=2, label='Total Local Units', color='#A23B72')
    
    ax1.set_title(f'Overall Evolution of Enterprises and Local Units ({year_threshold}+)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Growth rates
    ax2.plot(yearly_totals['Year'][1:], yearly_totals['Enterprises_growth'][1:], 
             marker='o', linewidth=2, label='Enterprises Growth %', color='#2E86AB')
    ax2.plot(yearly_totals['Year'][1:], yearly_totals['Local_units_growth'][1:], 
             marker='s', linewidth=2, label='Local Units Growth %', color='#A23B72')
    
    ax2.set_title('Year-over-Year Growth Rates', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Growth Rate (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nOverall Growth Summary:")
    print(f"  Total Enterprises: {yearly_totals['Enterprises'].iloc[-1]:,.0f}")
    print(f"  Total Local Units: {yearly_totals['Local units'].iloc[-1]:,.0f}")
    print(f"  Avg Annual Growth (Enterprises): {yearly_totals['Enterprises_growth'].mean():.2f}%")
    print(f"  Avg Annual Growth (Local Units): {yearly_totals['Local_units_growth'].mean():.2f}%")
    
    return yearly_totals


# ============================================================================
# MAIN RUNNER FUNCTION
# ============================================================================

def run_comprehensive_analysis(filepath='processed_trade_data.csv'):
    """
    Run all 5 core business analyses on the trade data
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing trade data
        
    Returns:
    --------
    results : dict
        Dictionary containing results from all analyses
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE BUSINESS ANALYSIS DASHBOARD")
    print("="*70)
    
    # Setup
    setup_plotting_style()
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    
    results = {}
    
    # Run all analyses
    try:
        results['profitability'] = analyze_profitability_trends(df)
    except Exception as e:
        print(f"Error in profitability analysis: {e}")
    
    try:
        results['capital_formation'] = analyze_capital_formation_trends(df)
    except Exception as e:
        print(f"Error in capital formation analysis: {e}")
    
    try:
        results['labor_productivity'] = analyze_labor_productivity_by_size(df)
    except Exception as e:
        print(f"Error in labor productivity analysis: {e}")
    
    try:
        results['size_productivity'] = analyze_size_productivity_relationship(df)
    except Exception as e:
        print(f"Error in size-productivity analysis: {e}")
    
    try:
        results['growth_patterns'] = analyze_enterprise_growth_patterns(df)
    except Exception as e:
        print(f"Error in growth patterns analysis: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results
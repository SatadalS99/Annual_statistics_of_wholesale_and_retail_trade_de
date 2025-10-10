# German Trade and Economic Activity Analysis (2005–Present)

## Overview
This project explores structural dynamics and performance patterns within the German economy by analyzing trade data spanning from 2005 onward. Using official statistics classified according to WZ2008 (Wirtschaftszweig 2008) standards, the analysis investigates profitability trends, capital allocation, operational efficiency, firm characteristics, and temporal evolution across different economic sectors and firm sizes.

**Disclaimer**: This project is conducted solely for academic and analytical purposes. All interpretations are based on publicly available statistical data and should not be considered financial advice, investment guidance, or professional economic assessment.

## Research Questions
The analysis focuses on the following five research questions:

1. **Profitability Trends Analysis**: How has the gross yield ratio changed across industries over time, and which industries demonstrate the most stable or volatile profitability?

2. **Sectoral Capital Dynamics**: Which economic activities exhibit the highest gross capital formation relative to turnover, and how has this ratio changed over time?

3. **Labor Utilization Efficiency**: How does turnover per person employed vary across different employee size classes and sectors?

4. **Firm Size and Productivity**: What is the relationship between employee size class and turnover per enterprise across industries?

5. **Temporal Growth Patterns**: How have the number of enterprises and local units evolved across different economic activities from 2005 onward?

## Getting Started

### Prerequisites
To run this project, you need:
* Python 3.8+
* pandas
* numpy
* matplotlib
* seaborn
* openpyxl (for Excel file processing)

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

### Installation
1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/german-trade-analysis.git
cd german-trade-analysis
```

2. Place the datasets inside the `data/` folder:
   - `processed_trade_data.csv` (cleaned dataset)
   - `Unprocessed_trade_data.xlsx` (original dataset)

3. Navigate to the `src/` folder and open the analysis notebooks.

## Usage
Run the analysis notebooks from top to bottom to:
* Load and explore the processed trade data
* Apply statistical analysis and visualization functions
* Generate insights on:
  - Industry profitability patterns and volatility
  - Capital investment trends across sectors
  - Labor productivity by firm size and industry
  - Economies of scale analysis
  - Long-term growth trajectories

Reusable functions for data processing and analysis are available in `src/analysis_utils.py`.

### Project Structure
```
├── data/
│   ├── processed_trade_data.csv
│   └── Unprocessed_trade_data.xlsx
├── src/
│   ├── analysis_notebooks.ipynb
│   └── analysis_utils.py
├── results/
│   ├── figures/
│   └── reports/
├── README.md
└── requirements.txt
```

## Data

### Source
* **Origin**: German Federal Statistical Office ([Destatis](https://www-genesis.destatis.de/datenbank/online/statistic/45341/table/45341-0002))
* **Classification**: WZ2008 (Wirtschaftszweig 2008) - German classification of economic activities aligned with European NACE Rev. 2

### Dataset Characteristics
* **Format**: CSV (processed) and XLSX (unprocessed)
* **Temporal Coverage**: 2005 onwards
* **Total Observations**: 8,064 rows
* **Granularity**: Year × Economic Activity × Employee Size Class

### Variables (15 columns)
* **Identifiers**: Year, WZ2008_Code, Economic_Activity, Employee_Size_Class
* **Enterprise Metrics**: Enterprises, Local units, Persons employed
* **Financial Metrics**: Turnover, Turnover per enterprise, Turnover per person employed
* **Operational Metrics**: Input of goods, Gross capital formation, Disposal of tangible fixed assets, Gross yield ratio, Expenditure

## Key Features
* **Multi-dimensional Analysis**: Combines industry, firm size, and temporal dimensions
* **Longitudinal Perspective**: Tracks economic patterns across nearly two decades
* **Comprehensive Metrics**: Covers profitability, productivity, capital investment, and growth
* **Cleaned Data**: Standardized format ready for immediate analysis
* **Reusable Code**: Modular functions for reproducible research

## Results
The analysis provides actionable insights for:
* **Policymakers**: Understanding sectoral performance and capital allocation
* **Investors**: Identifying stable, profitable sectors and growth opportunities
* **Business Leaders**: Benchmarking performance against industry standards
* **Researchers**: Contributing empirical evidence to industrial economics discussions

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaboration, please contact:

**[Satadal Santra]**  
Email: satadals121@gmail.com  
GitHub: ([https://github.com/SatadalS99](https://github.com/SatadalS99))

## Acknowledgments
* Data provided by the German Federal Statistical Office (Destatis)
* WZ2008 classification system based on European NACE Rev. 2 standards

---

**Citation**: If you use this analysis or methodology in your research, please cite this repository appropriately.

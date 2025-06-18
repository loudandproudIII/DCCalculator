#!/usr/bin/env python3
"""
Generate comprehensive thermal analysis template XLSX file
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_thermal_analysis_template():
    """Create populated template file with realistic examples"""
    
    # Define column structure with descriptions
    columns_info = {
        # Basic Configuration
        'Case_Name': 'Descriptive name for the analysis case',
        'chip_config': 'Chip configuration key (e.g., standard_gpu, high_performance_gpu)',
        
        # Power Configuration (Option 1: Fixed Power)
        'Power_W': 'Fixed power dissipation in Watts (leave blank for power model)',
        
        # Power Configuration (Option 2: Power Model)
        'V_dd': 'Supply voltage in Volts (for power model)',
        'f_clock_GHz': 'Clock frequency in GHz (for power model)',
        'C_eff': 'Effective capacitance in Farads (default: 1e-9)',
        'alpha': 'Activity factor 0-1 (default: 0.7)',
        'cores': 'Number of cores (default: 1)',
        
        # Environmental
        'T_ambient_C': 'Ambient temperature in Celsius',
        
        # Cooling Configuration
        'cooling_type': 'Cooling type: air, liquid, evaporative, vapor_chamber',
        
        # Air Cooling Parameters
        'h_air': 'Air heat transfer coefficient W/m²·K',
        'A_sink_cm2': 'Heatsink area in cm²',
        
        # Liquid Cooling Parameters
        'h_liquid': 'Liquid heat transfer coefficient W/m²·K',
        'A_cold_m2': 'Cold plate area in m²',
        'T_coolant_C': 'Coolant temperature in Celsius',
        
        # Evaporative Cooling Parameters
        'fluid': 'Working fluid (e.g., water, r134a)',
        'pressure_bar': 'System pressure in bar',
        
        # Vapor Chamber Parameters
        'vc_config': 'Vapor chamber configuration',
        'T_condenser_C': 'Condenser temperature in Celsius',
        
        # Custom Layers (Optional)
        'Layer1_Name': 'Custom layer 1 name',
        'Layer1_Thickness_mm': 'Custom layer 1 thickness in mm',
        'Layer1_Area_m2': 'Custom layer 1 area in m²',
        'Layer1_Material': 'Custom layer 1 material key',
        
        'Layer2_Name': 'Custom layer 2 name',
        'Layer2_Thickness_mm': 'Custom layer 2 thickness in mm',
        'Layer2_Area_m2': 'Custom layer 2 area in m²',
        'Layer2_Material': 'Custom layer 2 material key',
        
        'Layer3_Name': 'Custom layer 3 name',
        'Layer3_Thickness_mm': 'Custom layer 3 thickness in mm',
        'Layer3_Area_m2': 'Custom layer 3 area in m²',
        'Layer3_Material': 'Custom layer 3 material key',
        
        # Advanced Options
        'max_iterations': 'Max iterations for nonlinear solver',
        'convergence_tolerance_K': 'Temperature convergence tolerance',
        
        # Comments
        'Notes': 'Additional notes or comments'
    }
    
    # Create main data
    data = []
    
    # Example 1: Entry-level GPU with air cooling (fixed power)
    data.append({
        'Case_Name': 'Entry GPU - Air Cooled',
        'chip_config': 'standard_gpu',
        'Power_W': 75,
        'T_ambient_C': 25,
        'cooling_type': 'air',
        'h_air': 30,
        'A_sink_cm2': 50,
        'Notes': 'Low-power GPU with modest air cooling'
    })
    
    # Example 2: Mid-range GPU with better air cooling
    data.append({
        'Case_Name': 'Mid-range GPU - High Airflow',
        'chip_config': 'standard_gpu',
        'Power_W': 150,
        'T_ambient_C': 28,
        'cooling_type': 'air',
        'h_air': 80,
        'A_sink_cm2': 150,
        'Notes': 'Typical gaming GPU with good air cooler'
    })
    
    # Example 3: High-end GPU with liquid cooling
    data.append({
        'Case_Name': 'High-end GPU - AIO Liquid',
        'chip_config': 'high_performance_gpu',
        'Power_W': 300,
        'T_ambient_C': 24,
        'cooling_type': 'liquid',
        'h_liquid': 5000,
        'A_cold_m2': 0.002,
        'T_coolant_C': 20,
        'Notes': 'Premium GPU with 240mm AIO cooler'
    })
    
    # Example 4: Power model example - overclocked GPU
    data.append({
        'Case_Name': 'Overclocked GPU - Dynamic Power',
        'chip_config': 'high_performance_gpu',
        'V_dd': 1.2,
        'f_clock_GHz': 2.5,
        'C_eff': 1.2e-9,
        'alpha': 0.8,
        'cores': 3584,
        'T_ambient_C': 30,
        'cooling_type': 'liquid',
        'h_liquid': 8000,
        'A_cold_m2': 0.0025,
        'T_coolant_C': 18,
        'max_iterations': 100,
        'convergence_tolerance_K': 0.5,
        'Notes': 'Overclocked with power model, high-performance liquid cooling'
    })
    
    # Example 5: Data center GPU with evaporative cooling
    data.append({
        'Case_Name': 'Data Center GPU - Evaporative',
        'chip_config': 'high_performance_gpu',
        'Power_W': 400,
        'T_ambient_C': 35,
        'cooling_type': 'evaporative',
        'fluid': 'dielectric_fluid',
        'pressure_bar': 1.0,
        'Notes': 'High ambient temperature, two-phase cooling'
    })
    
    # Example 6: Vapor chamber cooling
    data.append({
        'Case_Name': 'Gaming Laptop - Vapor Chamber',
        'chip_config': 'standard_gpu',
        'Power_W': 120,
        'T_ambient_C': 30,
        'cooling_type': 'vapor_chamber',
        'vc_config': 'ultra_thin_vc',
        'T_condenser_C': 40,
        'Notes': 'Thin vapor chamber for laptop GPU'
    })
    
    # Example 7: Custom layer stack - direct die cooling
    data.append({
        'Case_Name': 'Custom Stack - Direct Die',
        'Power_W': 200,
        'T_ambient_C': 22,
        'cooling_type': 'liquid',
        'h_liquid': 10000,
        'A_cold_m2': 0.0004,
        'T_coolant_C': 15,
        'Layer1_Name': 'GPU_Die',
        'Layer1_Thickness_mm': 0.775,
        'Layer1_Area_m2': 0.0004,
        'Layer1_Material': 'silicon',
        'Layer2_Name': 'Direct_TIM',
        'Layer2_Thickness_mm': 0.02,
        'Layer2_Area_m2': 0.0004,
        'Layer2_Material': 'liquid_metal_tim',
        'Notes': 'Direct die cooling with liquid metal TIM'
    })
    
    # Example 8: Hot environment stress test
    data.append({
        'Case_Name': 'Stress Test - Hot Environment',
        'chip_config': 'standard_gpu',
        'Power_W': 250,
        'T_ambient_C': 45,
        'cooling_type': 'air',
        'h_air': 50,
        'A_sink_cm2': 100,
        'Notes': 'Testing thermal limits in hot environment'
    })
    
    # Example 9: Error case - invalid configuration
    data.append({
        'Case_Name': 'ERROR DEMO - Invalid Config',
        'chip_config': 'nonexistent_config',
        'Power_W': 150,
        'T_ambient_C': 25,
        'cooling_type': 'air',
        'h_air': 50,
        'A_sink_cm2': 100,
        'Notes': 'This will fail - demonstrates error handling'
    })
    
    # Example 10: Error case - out of range power
    data.append({
        'Case_Name': 'ERROR DEMO - Extreme Power',
        'chip_config': 'standard_gpu',
        'Power_W': 5000,
        'T_ambient_C': 25,
        'cooling_type': 'liquid',
        'h_liquid': 5000,
        'A_cold_m2': 0.002,
        'T_coolant_C': 20,
        'Notes': 'This will fail - power out of valid range'
    })
    
    # Example 11: Refrigerant cooling (if REFPROP available)
    data.append({
        'Case_Name': 'Advanced - R134a Refrigerant',
        'chip_config': 'high_performance_gpu',
        'Power_W': 350,
        'T_ambient_C': 25,
        'cooling_type': 'evaporative',
        'fluid': 'r134a',
        'pressure_bar': 5.0,
        'Notes': 'Refrigerant cooling - requires REFPROP'
    })
    
    # Example 12: Multi-chip module with custom stack
    data.append({
        'Case_Name': 'MCM - Custom Stack',
        'V_dd': 1.1,
        'f_clock_GHz': 2.0,
        'C_eff': 2e-9,
        'alpha': 0.75,
        'cores': 7168,
        'T_ambient_C': 26,
        'cooling_type': 'liquid',
        'h_liquid': 7000,
        'A_cold_m2': 0.003,
        'T_coolant_C': 22,
        'Layer1_Name': 'MCM_Dies',
        'Layer1_Thickness_mm': 0.775,
        'Layer1_Area_m2': 0.0009,
        'Layer1_Material': 'silicon',
        'Layer2_Name': 'Substrate',
        'Layer2_Thickness_mm': 1.2,
        'Layer2_Area_m2': 0.0016,
        'Layer2_Material': 'aluminum',
        'Layer3_Name': 'TIM',
        'Layer3_Thickness_mm': 0.05,
        'Layer3_Area_m2': 0.0016,
        'Layer3_Material': 'tim_high_performance',
        'Notes': 'Multi-chip module with custom substrate'
    })
    
    # Create main DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to match documentation order
    column_order = list(columns_info.keys())
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    # Create description DataFrame
    desc_data = []
    for col, desc in columns_info.items():
        desc_data.append({
            'Column': col,
            'Description': desc,
            'Required': 'Yes' if col in ['cooling_type', 'T_ambient_C'] else 'Conditional',
            'Example': 'See data rows'
        })
    
    descriptions_df = pd.DataFrame(desc_data)
    
    # Create units reference
    units_data = [
        {'Parameter': 'Power', 'Symbol': 'P', 'Unit': 'W', 'Notes': 'Watts'},
        {'Parameter': 'Temperature', 'Symbol': 'T', 'Unit': 'K or °C', 'Notes': 'Kelvin or Celsius as specified'},
        {'Parameter': 'Heat Transfer Coefficient', 'Symbol': 'h', 'Unit': 'W/m²·K', 'Notes': 'HTC'},
        {'Parameter': 'Area', 'Symbol': 'A', 'Unit': 'm² or cm²', 'Notes': 'As specified in column name'},
        {'Parameter': 'Thickness', 'Symbol': 't', 'Unit': 'm or mm', 'Notes': 'As specified in column name'},
        {'Parameter': 'Thermal Resistance', 'Symbol': 'R', 'Unit': 'K/W', 'Notes': 'Temperature rise per watt'},
        {'Parameter': 'Voltage', 'Symbol': 'V', 'Unit': 'V', 'Notes': 'Volts'},
        {'Parameter': 'Frequency', 'Symbol': 'f', 'Unit': 'Hz or GHz', 'Notes': 'As specified'},
        {'Parameter': 'Capacitance', 'Symbol': 'C', 'Unit': 'F', 'Notes': 'Farads'},
        {'Parameter': 'Pressure', 'Symbol': 'P', 'Unit': 'Pa or bar', 'Notes': 'As specified'}
    ]
    units_df = pd.DataFrame(units_data)
    
    # Material reference
    materials_data = [
        {'Material_Key': 'silicon', 'Description': 'Silicon semiconductor', 'k (W/m·K)': 130, 'Category': 'semiconductor'},
        {'Material_Key': 'copper', 'Description': 'Copper metal', 'k (W/m·K)': 385, 'Category': 'metal'},
        {'Material_Key': 'aluminum', 'Description': 'Aluminum metal', 'k (W/m·K)': 205, 'Category': 'metal'},
        {'Material_Key': 'tim_standard', 'Description': 'Standard thermal interface material', 'k (W/m·K)': 5, 'Category': 'interface_material'},
        {'Material_Key': 'tim_high_performance', 'Description': 'High-performance TIM', 'k (W/m·K)': 12, 'Category': 'interface_material'},
        {'Material_Key': 'liquid_metal_tim', 'Description': 'Liquid metal TIM', 'k (W/m·K)': 73, 'Category': 'interface_material'},
        {'Material_Key': 'solder', 'Description': 'Solder (SnAgCu)', 'k (W/m·K)': 57, 'Category': 'interface_material'}
    ]
    materials_df = pd.DataFrame(materials_data)
    
    # Cooling reference
    cooling_data = [
        {'Cooling_Type': 'air', 'h_typical (W/m²·K)': '25-250', 'Parameters': 'h_air, A_sink', 'Notes': 'Natural to forced convection'},
        {'Cooling_Type': 'liquid', 'h_typical (W/m²·K)': '1000-10000', 'Parameters': 'h_liquid, A_cold, T_coolant', 'Notes': 'Single-phase liquid cooling'},
        {'Cooling_Type': 'evaporative', 'h_typical (W/m²·K)': '10000-100000', 'Parameters': 'fluid, pressure', 'Notes': 'Two-phase boiling/evaporation'},
        {'Cooling_Type': 'vapor_chamber', 'h_typical (W/m²·K)': 'N/A', 'Parameters': 'vc_config, T_condenser', 'Notes': 'Heat spreading device'}
    ]
    cooling_df = pd.DataFrame(cooling_data)
    
    # Write to Excel with multiple sheets
    output_file = 'thermal_analysis_template.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Thermal_Analysis_Cases', index=False)
        
        # Column descriptions
        descriptions_df.to_excel(writer, sheet_name='Column_Descriptions', index=False)
        
        # Units reference
        units_df.to_excel(writer, sheet_name='Units_Reference', index=False)
        
        # Materials reference
        materials_df.to_excel(writer, sheet_name='Materials_Reference', index=False)
        
        # Cooling reference
        cooling_df.to_excel(writer, sheet_name='Cooling_Reference', index=False)
        
        # Format the main sheet
        workbook = writer.book
        worksheet = workbook['Thermal_Analysis_Cases']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Created template file: {output_file}")
    print(f"  - {len(df)} example cases")
    print(f"  - {len(descriptions_df)} documented columns")
    print(f"  - Multiple reference sheets")
    print("\nTemplate includes:")
    print("  - Simple steady-state examples")
    print("  - Dynamic power model examples")
    print("  - Various cooling methods")
    print("  - Custom layer stack examples")
    print("  - Error cases for testing")
    
    return output_file


if __name__ == "__main__":
    create_thermal_analysis_template()
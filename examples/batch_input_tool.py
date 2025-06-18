#!/usr/bin/env python3
"""
Batch Thermal Analysis Processor
Processes multiple thermal analysis cases from CSV/XLSX files
"""

import sys
import os
import pandas as pd
import numpy as np
import argparse
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main.utility.data_puller import DataPuller, LayerInfo
from main.calculation_hx.chip_junction_to_surf import (
    JunctionToSurfaceCalculator, CoolingType, PowerModel, ThermalLayer
)

# Try to import REFPROP interface
try:
    from main.utility.refprop_interface import REFPROP_AVAILABLE
except:
    REFPROP_AVAILABLE = False


class BatchThermalProcessor:
    """Batch processor for thermal analysis cases"""
    
    def __init__(self):
        self.data_puller = DataPuller()
        self.results = []
        self.errors = []
        
    def process_file(self, input_path: str) -> bool:
        """Process input file and write results"""
        print(f"Processing file: {input_path}")
        
        # Determine file type and read data
        file_ext = os.path.splitext(input_path)[1].lower()
        
        try:
            if file_ext == '.xlsx':
                df = pd.read_excel(input_path, sheet_name=0)
                is_excel = True
            elif file_ext == '.csv':
                df = pd.read_csv(input_path)
                is_excel = False
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
            print(f"Loaded {len(df)} rows from {file_ext} file")
            
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
        
        # Process each row
        results_data = []
        
        for idx, row in df.iterrows():
            print(f"\nProcessing row {idx + 1}/{len(df)}...")
            result = self.process_row(row, idx + 1)
            results_data.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Combine with original data
        output_df = pd.concat([df, results_df], axis=1)
        
        # Write results
        if is_excel:
            # For Excel, create new sheet
            output_path = input_path.replace('.xlsx', '_results.xlsx')
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Write original data to first sheet
                df.to_excel(writer, sheet_name='Input', index=False)
                # Write combined results to second sheet
                output_df.to_excel(writer, sheet_name='Results', index=False)
                # Write summary to third sheet
                summary_df = self.create_summary(results_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        else:
            # For CSV, append columns
            output_path = input_path.replace('.csv', '_results.csv')
            output_df.to_csv(output_path, index=False)
        
        print(f"\nResults written to: {output_path}")
        print(f"Processed {len(df)} cases:")
        print(f"  - Successful: {sum(1 for r in results_data if r['ERROR'] == '')}")
        print(f"  - Failed: {sum(1 for r in results_data if r['ERROR'] != '')}")
        
        return True
    
    def process_row(self, row: pd.Series, row_num: int) -> Dict[str, Any]:
        """Process a single row of thermal analysis"""
        result = {
            'Row': row_num,
            'Status': 'FAILED',
            'T_junction_K': np.nan,
            'T_junction_C': np.nan,
            'T_case_K': np.nan,
            'T_case_C': np.nan,
            'R_total_K/W': np.nan,
            'R_conduction_K/W': np.nan,
            'R_cooling_K/W': np.nan,
            'Power_total_W': np.nan,
            'Power_dynamic_W': np.nan,
            'Power_leakage_W': np.nan,
            'Convergence_iterations': np.nan,
            'ERROR': '',
            'Warnings': ''
        }
        
        try:
            # Parse configuration
            config = self.parse_row_config(row)
            
            # Validate configuration
            validation_errors = self.validate_config(config)
            if validation_errors:
                result['ERROR'] = '; '.join(validation_errors)
                return result
            
            # Create calculator
            calc = JunctionToSurfaceCalculator(self.data_puller)
            
            # Set up layer stack
            if config['use_custom_layers']:
                self.setup_custom_layers(calc, config['custom_layers'])
            else:
                calc.load_configuration(config['chip_config'])
            
            # Run analysis based on power model type
            if config.get('use_power_model', False):
                # Create power model
                power_model = PowerModel(
                    V_dd=config['V_dd'],
                    f_clock=config['f_clock'],
                    C_eff=config.get('C_eff', 1e-9),
                    alpha=config.get('alpha', 0.7),
                    cores=config.get('cores', 1)
                )
                
                # Solve nonlinear problem
                analysis_results = calc.solve_nonlinear(
                    power_model,
                    config['T_ambient'],
                    config['cooling_type'],
                    config['cooling_params'],
                    max_iter=config.get('max_iterations', 50),
                    tol=config.get('convergence_tol', 0.1)
                )
                
                # Extract results
                result['Power_dynamic_W'] = analysis_results.get('power_dynamic', np.nan)
                result['Power_leakage_W'] = analysis_results.get('power_leakage', np.nan)
                result['Convergence_iterations'] = analysis_results.get('iterations', np.nan)
                
                if not analysis_results.get('converged', False):
                    result['Warnings'] = 'Did not converge'
                    
            else:
                # Simple steady-state analysis
                analysis_results = calc.calculate_steady_state(
                    config['power_watts'],
                    config['T_ambient'],
                    config['cooling_type'],
                    config['cooling_params']
                )
            
            # Extract common results
            result['Status'] = 'SUCCESS'
            result['T_junction_K'] = analysis_results['temperatures']['T_junction']
            result['T_junction_C'] = result['T_junction_K'] - 273.15
            result['T_case_K'] = analysis_results['temperatures']['T_case']
            result['T_case_C'] = result['T_case_K'] - 273.15
            result['R_total_K/W'] = analysis_results['resistances']['R_total']
            result['R_cooling_K/W'] = analysis_results['resistances'].get('R_cooling', np.nan)
            result['Power_total_W'] = analysis_results['power']
            
            # Calculate conduction resistance
            result['R_conduction_K/W'] = result['R_total_K/W'] - result['R_cooling_K/W']
            
            # Add warnings for high temperatures
            if result['T_junction_C'] > 105:
                result['Warnings'] += 'CRITICAL: T_junction > 105°C; '
            elif result['T_junction_C'] > 95:
                result['Warnings'] += 'WARNING: T_junction > 95°C; '
                
        except Exception as e:
            result['ERROR'] = f"Analysis error: {str(e)}"
            print(f"  Error: {e}")
            # Uncomment for debugging:
            # traceback.print_exc()
        
        return result
    
    def parse_row_config(self, row: pd.Series) -> Dict[str, Any]:
        """Parse configuration from a data row"""
        config = {}
        
        # Basic parameters
        config['T_ambient'] = self.get_value(row, ['T_ambient_K', 'Ambient_Temperature_K'], 298.15)
        
        # Check for Celsius input
        if pd.notna(row.get('T_ambient_C')):
            config['T_ambient'] = row['T_ambient_C'] + 273.15
        
        # Power configuration
        if pd.notna(row.get('Power_W')) or pd.notna(row.get('power_watts')):
            # Fixed power mode
            config['power_watts'] = self.get_value(row, ['Power_W', 'power_watts'], 150)
            config['use_power_model'] = False
        else:
            # Power model mode
            config['use_power_model'] = True
            config['V_dd'] = self.get_value(row, ['V_dd', 'Supply_Voltage_V'], 1.1)
            config['f_clock'] = self.get_value(row, ['f_clock_Hz', 'Clock_Frequency_Hz'], 2e9)
            
            # Check for GHz input
            if pd.notna(row.get('f_clock_GHz')):
                config['f_clock'] = row['f_clock_GHz'] * 1e9
                
            config['C_eff'] = self.get_value(row, ['C_eff', 'Effective_Capacitance_F'], 1e-9)
            config['alpha'] = self.get_value(row, ['alpha', 'Activity_Factor'], 0.7)
            config['cores'] = int(self.get_value(row, ['cores', 'Num_Cores'], 1))
        
        # Chip configuration
        chip_config = self.get_value(row, ['chip_config', 'Chip_Configuration'], 'standard_gpu')
        config['chip_config'] = chip_config
        
        # Check for custom layers
        config['use_custom_layers'] = False
        config['custom_layers'] = []
        
        # Look for custom layer definitions
        layer_num = 1
        while True:
            layer_name_col = f'Layer{layer_num}_Name'
            if layer_name_col not in row or pd.isna(row[layer_name_col]):
                break
                
            layer_data = {
                'name': row[layer_name_col],
                'thickness': self.get_value(row, [f'Layer{layer_num}_Thickness_m'], 0.001),
                'area': self.get_value(row, [f'Layer{layer_num}_Area_m2'], 0.0004),
                'material': self.get_value(row, [f'Layer{layer_num}_Material'], 'silicon')
            }
            
            # Check for mm input
            if pd.notna(row.get(f'Layer{layer_num}_Thickness_mm')):
                layer_data['thickness'] = row[f'Layer{layer_num}_Thickness_mm'] / 1000
                
            config['custom_layers'].append(layer_data)
            config['use_custom_layers'] = True
            layer_num += 1
        
        # Cooling configuration
        cooling_type_str = self.get_value(row, ['cooling_type', 'Cooling_Type'], 'air')
        config['cooling_type'] = self.parse_cooling_type(cooling_type_str)
        
        # Parse cooling parameters based on type
        config['cooling_params'] = self.parse_cooling_params(row, config['cooling_type'])
        
        # Advanced options
        config['max_iterations'] = int(self.get_value(row, ['max_iterations'], 50))
        config['convergence_tol'] = self.get_value(row, ['convergence_tolerance_K'], 0.1)
        
        return config
    
    def get_value(self, row: pd.Series, columns: List[str], default: Any) -> Any:
        """Get value from row, checking multiple possible column names"""
        for col in columns:
            if col in row and pd.notna(row[col]):
                return row[col]
        return default
    
    def parse_cooling_type(self, cooling_str: str) -> CoolingType:
        """Parse cooling type from string"""
        cooling_map = {
            'air': CoolingType.AIR,
            'liquid': CoolingType.LIQUID,
            'evaporative': CoolingType.EVAPORATIVE,
            'vapor_chamber': CoolingType.VAPOR_CHAMBER,
            'vapor chamber': CoolingType.VAPOR_CHAMBER,
            'vc': CoolingType.VAPOR_CHAMBER,
            'hybrid': CoolingType.HYBRID
        }
        
        cooling_lower = cooling_str.lower().strip()
        if cooling_lower in cooling_map:
            return cooling_map[cooling_lower]
        else:
            raise ValueError(f"Unknown cooling type: {cooling_str}")
    
    def parse_cooling_params(self, row: pd.Series, cooling_type: CoolingType) -> Dict[str, Any]:
        """Parse cooling parameters based on cooling type"""
        params = {}
        
        if cooling_type == CoolingType.AIR:
            params['h_air'] = self.get_value(row, ['h_air', 'h_air_W/m2K'], 50)
            params['A_sink'] = self.get_value(row, ['A_sink_m2', 'Heatsink_Area_m2'], 0.01)
            
            # Check for cm² input
            if pd.notna(row.get('A_sink_cm2')):
                params['A_sink'] = row['A_sink_cm2'] / 10000
                
        elif cooling_type == CoolingType.LIQUID:
            params['h_liquid'] = self.get_value(row, ['h_liquid', 'h_liquid_W/m2K'], 5000)
            params['A_cold'] = self.get_value(row, ['A_cold_m2', 'Cold_Plate_Area_m2'], 0.0016)
            params['T_coolant'] = self.get_value(row, ['T_coolant_K', 'Coolant_Temperature_K'], 288)
            
            # Check for Celsius input
            if pd.notna(row.get('T_coolant_C')):
                params['T_coolant'] = row['T_coolant_C'] + 273.15
                
        elif cooling_type == CoolingType.EVAPORATIVE:
            params['fluid'] = self.get_value(row, ['fluid', 'Working_Fluid'], 'water')
            params['pressure'] = self.get_value(row, ['pressure_Pa', 'System_Pressure_Pa'], 101325)
            params['use_refprop'] = self.get_value(row, ['use_refprop'], False)
            
            # Check for bar input
            if pd.notna(row.get('pressure_bar')):
                params['pressure'] = row['pressure_bar'] * 1e5
                
        elif cooling_type == CoolingType.VAPOR_CHAMBER:
            params['vc_config'] = self.get_value(row, ['vc_config', 'VC_Configuration'], 'standard_vc')
            params['T_condenser'] = self.get_value(row, ['T_condenser_K', 'Condenser_Temperature_K'], 298)
            
            # Check for Celsius input
            if pd.notna(row.get('T_condenser_C')):
                params['T_condenser'] = row['T_condenser_C'] + 273.15
                
        elif cooling_type == CoolingType.HYBRID:
            # Simplified hybrid - just use liquid cooling params
            params['h_liquid'] = self.get_value(row, ['h_liquid'], 10000)
            params['A_cold'] = self.get_value(row, ['A_cold_m2'], 0.002)
            params['T_coolant'] = self.get_value(row, ['T_coolant_K'], 288)
        
        return params
    
    def setup_custom_layers(self, calc: JunctionToSurfaceCalculator, layers_data: List[Dict]):
        """Set up custom layer stack"""
        calc.layers = []
        
        for layer_data in layers_data:
            layer_info = LayerInfo(
                name=layer_data['name'],
                thickness=layer_data['thickness'],
                area=layer_data['area'],
                material_key=layer_data['material'],
                notes="Custom layer from batch input"
            )
            
            material = calc.data.get_material(layer_data['material'])
            calc.layers.append(ThermalLayer(layer_info, material))
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Temperature validation
        if config['T_ambient'] < 200 or config['T_ambient'] > 400:
            errors.append(f"Ambient temperature {config['T_ambient']}K out of range (200-400K)")
        
        # Power validation
        if config.get('use_power_model'):
            if config['V_dd'] < 0.5 or config['V_dd'] > 2.0:
                errors.append(f"Supply voltage {config['V_dd']}V out of range (0.5-2.0V)")
            if config['f_clock'] < 1e8 or config['f_clock'] > 10e9:
                errors.append(f"Clock frequency {config['f_clock']/1e9:.1f}GHz out of range (0.1-10GHz)")
        else:
            if config['power_watts'] < 1 or config['power_watts'] > 1000:
                errors.append(f"Power {config['power_watts']}W out of range (1-1000W)")
        
        # Chip configuration validation
        if not config['use_custom_layers']:
            available_configs = self.data_puller.list_available_configurations()
            if config['chip_config'] not in available_configs:
                errors.append(f"Unknown chip configuration: {config['chip_config']}")
        
        # Material validation for custom layers
        if config['use_custom_layers']:
            available_materials = self.data_puller.list_available_materials()
            for i, layer in enumerate(config['custom_layers']):
                if layer['material'] not in available_materials:
                    errors.append(f"Unknown material '{layer['material']}' in layer {i+1}")
                if layer['thickness'] <= 0:
                    errors.append(f"Invalid thickness in layer {i+1}")
                if layer['area'] <= 0:
                    errors.append(f"Invalid area in layer {i+1}")
        
        # Cooling validation
        if config['cooling_type'] == CoolingType.EVAPORATIVE:
            if not REFPROP_AVAILABLE and config['cooling_params'].get('use_refprop'):
                errors.append("REFPROP not available for refrigerant calculations")
        
        return errors
    
    def create_summary(self, results_data: List[Dict]) -> pd.DataFrame:
        """Create summary statistics DataFrame"""
        summary = {
            'Metric': [],
            'Value': []
        }
        
        # Count successful/failed
        success_count = sum(1 for r in results_data if r['ERROR'] == '')
        fail_count = sum(1 for r in results_data if r['ERROR'] != '')
        
        summary['Metric'].extend(['Total Cases', 'Successful', 'Failed'])
        summary['Value'].extend([len(results_data), success_count, fail_count])
        
        # Temperature statistics
        t_junction_values = [r['T_junction_C'] for r in results_data if not np.isnan(r.get('T_junction_C', np.nan))]
        
        if t_junction_values:
            summary['Metric'].extend([
                'Min T_junction (°C)',
                'Max T_junction (°C)',
                'Mean T_junction (°C)',
                'Cases > 95°C',
                'Cases > 105°C'
            ])
            summary['Value'].extend([
                f"{min(t_junction_values):.1f}",
                f"{max(t_junction_values):.1f}",
                f"{np.mean(t_junction_values):.1f}",
                sum(1 for t in t_junction_values if t > 95),
                sum(1 for t in t_junction_values if t > 105)
            ])
        
        return pd.DataFrame(summary)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Batch Thermal Analysis Processor for GPU thermal calculations'
    )
    parser.add_argument(
        'input_file',
        help='Input CSV or XLSX file containing thermal analysis parameters'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)
    
    # Process file
    processor = BatchThermalProcessor()
    
    print("="*60)
    print("GPU THERMAL ANALYSIS BATCH PROCESSOR")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Suppress detailed output unless verbose
    if not args.verbose:
        sys.stdout = open(os.devnull, 'w')
        success = processor.process_file(args.input_file)
        sys.stdout = sys.__stdout__
    else:
        success = processor.process_file(args.input_file)
    
    if success:
        print("\nBatch processing completed successfully!")
    else:
        print("\nBatch processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
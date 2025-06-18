import sys
import os
# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Simple usage
from main.calculation_hx.chip_junction_to_surf import JunctionToSurfaceCalculator, CoolingType

# Create calculator
calc = JunctionToSurfaceCalculator()

# Load a configuration
calc.load_configuration('standard_gpu')

# Example 1: Liquid Cooling
print("Example 1: Liquid cooling:")
# Calculate temperatures
results = calc.calculate_steady_state(
    power=250,  # 250W
    T_ambient=298,  # 25°C
    cooling_type=CoolingType.LIQUID,
    cooling_params={'h_liquid': 10000, 'A_cold': 0.0016}
)

# Ensure the results dictionary contains the expected keys
if 'temperatures' in results and 'T_junction' in results['temperatures']:
    T_junction_celsius = results['temperatures']['T_junction'] - 273.15
    print(f"Junction Temperature: {T_junction_celsius:.1f}°C")
else:
    print("Error: Unexpected results format or missing data.")


# Example 2: Traditional evaporative cooling with water
print("Evaporative cooling with water:")
results = calc.calculate_steady_state(
    power=250,  # 250W
    T_ambient=298,  # 25°C
    cooling_type=CoolingType.EVAPORATIVE,
    cooling_params={'fluid': 'water', 'use_refprop': False}
)

if 'temperatures' in results and 'T_junction' in results['temperatures']:
    T_junction_celsius = results['temperatures']['T_junction'] - 273.15
    print(f"Junction Temperature: {T_junction_celsius:.1f}°C")

# Example 3: Refrigerant cooling (requires REFPROP)
print("\nEvaporative cooling with R134a at 5 bar:")
try:
    results = calc.calculate_steady_state(
        power=250,  # 250W
        T_ambient=298,  # 25°C
        cooling_type=CoolingType.EVAPORATIVE,
        cooling_params={
            'fluid': 'r134a',
            'pressure': 500000,  # 5 bar in Pa
            'use_refprop': True
        }
    )
    
    if 'temperatures' in results and 'T_junction' in results['temperatures']:
        T_junction_celsius = results['temperatures']['T_junction'] - 273.15
        print(f"Junction Temperature: {T_junction_celsius:.1f}°C")
        
except Exception as e:
    print(f"Note: Refrigerant calculation requires REFPROP. Error: {e}")


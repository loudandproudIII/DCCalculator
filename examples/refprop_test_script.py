import ctREFPROP.ctREFPROP as RP

# Initialize REFPROP
RP = RP.REFPROPFunctionLibrary("C:/Program Files (x86)/REFPROP")  # Update path to your REFPROP installation
RP.SETPATHdll("C:/Program Files (x86)/REFPROP")

# Example: Calculate properties for water
fluid = "Water"
temperature = 300  # Kelvin
pressure = 101.325  # kPa

# Get density (kg/m³)
result = RP.REFPROPdll(fluid, "TP", "D", RP.MOLAR_BASE_SI, 0, 0, temperature, pressure, [1.0])
print(f"Density: {result.Output[0]} kg/m³")
#!/usr/bin/env python3
"""
Data Puller Module
Interfaces between thermal calculators and JSON data files
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MaterialProperties:
    """Material thermal properties"""
    name: str
    k: float  # Thermal conductivity [W/m·K]
    rho: float  # Density [kg/m³]
    cp: float  # Specific heat [J/kg·K]
    category: str
    k_temp_ref: float = 300  # Reference temperature for k
    k_temp_coeff: float = 0  # Temperature coefficient
    contact_resistance: Optional[float] = None  # Contact resistance [m²·K/W]
    
    def k_temp_dependent(self, T: float) -> float:
        """Calculate temperature-dependent thermal conductivity"""
        if self.k_temp_coeff > 0 and self.category == 'semiconductor':
            return self.k * (self.k_temp_ref / T) ** self.k_temp_coeff
        return self.k


@dataclass
class LayerInfo:
    """Layer geometric and material information"""
    name: str
    thickness: float  # [m]
    area: float  # [m²]
    material_key: str
    notes: str = ""


@dataclass
class FluidProperties:
    """Fluid thermal properties for cooling"""
    name: str
    rho_l: float  # Liquid density [kg/m³]
    rho_v: float  # Vapor density [kg/m³]
    h_fg: float  # Latent heat [J/kg]
    sigma: float  # Surface tension [N/m]
    mu_l: float  # Liquid viscosity [Pa·s]
    cp_l: float  # Liquid specific heat [J/kg·K]
    Pr_l: float  # Liquid Prandtl number
    T_sat: float  # Saturation temperature [K]
    k_l: float  # Liquid thermal conductivity [W/m·K]


class DataPuller:
    """Main data interface class"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize data puller with directory containing JSON files"""
        if data_dir is None:
            hxvals_dir = os.path.dirname("./data/hxvals/")
            materials_dir = os.path.dirname("./data/materials/")
            #data_dir = os.path.dirname(os.path.abspath(__file__))
        
        #self.data_dir = Path(data_dir)
        self.hxvals_dir = Path(hxvals_dir)
        self.materials_dir = Path(materials_dir)
        self.materials_file = self.materials_dir / "chip_mat.json"
        self.dimensions_file = self.hxvals_dir / "chip_dims.json"
        
        # Cache loaded data
        self._materials_data = None
        self._dimensions_data = None
        
        # Load data on initialization
        self._load_data()
    
    def _load_data(self):
        """Load JSON data files"""
        try:
            with open(self.materials_file, 'r') as f:
                self._materials_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Materials file not found: {self.materials_file}")
        
        try:
            with open(self.dimensions_file, 'r') as f:
                self._dimensions_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dimensions file not found: {self.dimensions_file}")
    
    def get_material(self, material_key: str) -> MaterialProperties:
        """Get material properties by key"""
        if material_key not in self._materials_data['materials']:
            raise ValueError(f"Material '{material_key}' not found in database")
        
        mat_data = self._materials_data['materials'][material_key]
        
        # Extract thermal conductivity value and temperature dependence
        k_data = mat_data['thermal_conductivity']
        k_value = k_data['value']
        k_temp_ref = k_data.get('temperature_ref', 300)
        k_temp_coeff = k_data.get('temperature_coefficient', 0)
        
        return MaterialProperties(
            name=mat_data['name'],
            k=k_value,
            rho=mat_data['density']['value'],
            cp=mat_data['specific_heat']['value'],
            category=mat_data.get('category', 'unknown'),
            k_temp_ref=k_temp_ref,
            k_temp_coeff=k_temp_coeff,
            contact_resistance=mat_data.get('contact_resistance', {}).get('value')
        )
    
    def get_fluid_properties(self, fluid_key: str) -> FluidProperties:
        """Get fluid properties by key"""
        if fluid_key not in self._materials_data['fluids']:
            raise ValueError(f"Fluid '{fluid_key}' not found in database")
        
        fluid_data = self._materials_data['fluids'][fluid_key]
        props = fluid_data['properties']
        
        return FluidProperties(
            name=fluid_data['name'],
            rho_l=props['density_liquid'],
            rho_v=props['density_vapor'],
            h_fg=props['latent_heat'],
            sigma=props['surface_tension'],
            mu_l=props['viscosity_liquid'],
            cp_l=props['specific_heat_liquid'],
            Pr_l=props['prandtl_liquid'],
            T_sat=props['saturation_temperature'],
            k_l=props.get('thermal_conductivity_liquid', 0.6)
        )
    
    def get_chip_configuration(self, config_key: str) -> List[LayerInfo]:
        """Get chip layer configuration"""
        if config_key not in self._dimensions_data['chip_configurations']:
            raise ValueError(f"Configuration '{config_key}' not found in database")
        
        config_data = self._dimensions_data['chip_configurations'][config_key]
        layers = []
        
        for layer_data in config_data['layers']:
            layers.append(LayerInfo(
                name=layer_data['name'],
                thickness=layer_data['thickness'],
                area=layer_data['area'],
                material_key=layer_data['material'],
                notes=layer_data.get('notes', '')
            ))
        
        return layers
    
    def get_vapor_chamber_config(self, config_key: str) -> Dict[str, Any]:
        """Get vapor chamber configuration"""
        if config_key not in self._dimensions_data['vapor_chamber_configs']:
            raise ValueError(f"Vapor chamber config '{config_key}' not found")
        
        return self._dimensions_data['vapor_chamber_configs'][config_key]
    
    def get_heatsink_dimensions(self, cooling_type: str, size: str) -> Dict[str, float]:
        """Get heatsink dimensions"""
        try:
            return self._dimensions_data['heatsink_dimensions'][cooling_type][size]
        except KeyError:
            raise ValueError(f"Heatsink config '{cooling_type}/{size}' not found")
    
    def get_cooling_htc_range(self, cooling_type: str, mode: str) -> Dict[str, Any]:
        """Get typical heat transfer coefficient ranges"""
        try:
            return self._materials_data['cooling_parameters'][cooling_type][mode]
        except KeyError:
            raise ValueError(f"Cooling parameters '{cooling_type}/{mode}' not found")
    
    def list_available_materials(self) -> List[str]:
        """List all available material keys"""
        return list(self._materials_data['materials'].keys())
    
    def list_available_configurations(self) -> List[str]:
        """List all available chip configurations"""
        return list(self._dimensions_data['chip_configurations'].keys())
    
    def list_available_fluids(self) -> List[str]:
        """List all available cooling fluids"""
        return list(self._materials_data['fluids'].keys())
    
    def get_surface_properties(self, surface_type: str) -> Dict[str, float]:
        """Get surface finish properties"""
        if surface_type not in self._dimensions_data['surface_properties']:
            raise ValueError(f"Surface type '{surface_type}' not found")
        
        return self._dimensions_data['surface_properties'][surface_type]
    
    def save_custom_material(self, key: str, material_data: Dict[str, Any]):
        """Save a custom material to the database"""
        self._materials_data['materials'][key] = material_data
        
        with open(self.materials_file, 'w') as f:
            json.dump(self._materials_data, f, indent=2)
    
    def save_custom_configuration(self, key: str, config_data: Dict[str, Any]):
        """Save a custom chip configuration"""
        self._dimensions_data['chip_configurations'][key] = config_data
        
        with open(self.dimensions_file, 'w') as f:
            json.dump(self._dimensions_data, f, indent=2)


# Convenience functions for quick access
_default_puller = None

def get_default_puller() -> DataPuller:
    """Get or create default data puller instance"""
    global _default_puller
    if _default_puller is None:
        _default_puller = DataPuller()
    return _default_puller

def get_material(material_key: str) -> MaterialProperties:
    """Quick access to material properties"""
    return get_default_puller().get_material(material_key)

def get_chip_config(config_key: str) -> List[LayerInfo]:
    """Quick access to chip configuration"""
    return get_default_puller().get_chip_configuration(config_key)

def get_fluid(fluid_key: str) -> FluidProperties:
    """Quick access to fluid properties"""
    return get_default_puller().get_fluid_properties(fluid_key)

def is_refrigerant(self, fluid_key: str) -> bool:
    """Check if a fluid is a refrigerant (requires REFPROP)"""
    refrigerants = [
        'r134a', 'r1234yf', 'r1234ze', 'r245fa', 'r410a', 'r32',
        'r404a', 'r507a', 'r22', 'r744', 'r717', 'r290', 'r600a',
        'co2', 'ammonia', 'propane', 'isobutane'
    ]
    return fluid_key.lower() in refrigerants

def get_fluid_or_refrigerant(self, fluid_key: str, pressure: Optional[float] = None) -> Any:
    """
    Get fluid properties from either JSON database or REFPROP
    
    Args:
        fluid_key: Fluid identifier
        pressure: System pressure [Pa] (for refrigerants)
        
    Returns:
        FluidProperties or RefrigerantProperties object
    """
    if self.is_refrigerant(fluid_key):
        # Try to use REFPROP for refrigerants
        try:
            from main.utility.refprop_interface import RefpropInterface, REFPROP_AVAILABLE
            if REFPROP_AVAILABLE:
                interface = RefpropInterface()
                interface.set_fluid(fluid_key)
                if pressure:
                    return interface.get_saturation_properties(P=pressure)
                else:
                    # Default to 1 bar if no pressure specified
                    return interface.get_saturation_properties(P=101325)
        except:
            pass
    
    # Fall back to JSON database
    return self.get_fluid_properties(fluid_key)

if __name__ == "__main__":
    # Test the data puller
    puller = DataPuller()
    
    print("Available materials:", puller.list_available_materials())
    print("\nAvailable configurations:", puller.list_available_configurations())
    print("\nAvailable fluids:", puller.list_available_fluids())
    
    # Test getting material
    silicon = puller.get_material('silicon')
    print(f"\nSilicon properties: k={silicon.k} W/m·K, ρ={silicon.rho} kg/m³")
    
    # Test getting configuration
    config = puller.get_chip_configuration('standard_gpu')
    print(f"\nStandard GPU has {len(config)} layers")
    for layer in config:
        print(f"  - {layer.name}: {layer.thickness*1000:.3f} mm thick, "
              f"area={layer.area*1e6:.1f} mm², material={layer.material_key}")
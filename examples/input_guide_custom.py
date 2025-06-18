#!/usr/bin/env python3
"""
Advanced Input Guide Tool for GPU Thermal Analysis
Interactive guide with granular customization capabilities
"""

import sys
import os
import copy
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main.utility.data_puller import DataPuller, LayerInfo, MaterialProperties
from main.calculation_hx.chip_junction_to_surf import CoolingType, JunctionToSurfaceCalculator, PowerModel, ThermalLayer


class MaterialEditor:
    """Dedicated class for material management"""
    
    def __init__(self, data_puller: DataPuller):
        self.data = data_puller
        self.custom_materials = {}
    
    def create_material(self) -> Tuple[str, Dict[str, Any]]:
        """Interactive material creation"""
        print("\n" + "="*50)
        print("CREATE CUSTOM MATERIAL")
        print("="*50)
        
        # Get material name
        while True:
            name = input("\nEnter material name: ").strip()
            if name:
                break
            print("❌ Material name cannot be empty")
        
        # Create unique key
        key = name.lower().replace(' ', '_')
        if key in self.data.list_available_materials():
            key = f"{key}_custom"
            print(f"ℹ️  Material key will be: {key}")
        
        print("\nEnter material properties:")
        
        # Thermal conductivity
        print("\n1. Thermal Conductivity")
        print("   Typical values:")
        print("   - Metals: 50-400 W/m·K")
        print("   - Semiconductors: 1-150 W/m·K")
        print("   - Polymers: 0.1-10 W/m·K")
        
        k = self._get_numeric_input("Thermal conductivity (W/m·K)", 0.01, 1000, None)
        
        # Check if temperature dependent
        temp_dependent = input("\nIs thermal conductivity temperature-dependent? (y/n) [n]: ").strip().lower() == 'y'
        k_temp_coeff = 0
        if temp_dependent:
            print("   For semiconductors, typically 0.5-1.5")
            k_temp_coeff = self._get_numeric_input("Temperature coefficient", 0, 2, 0.8)
        
        # Density
        print("\n2. Density")
        print("   Typical values:")
        print("   - Metals: 2000-20000 kg/m³")
        print("   - Semiconductors: 2000-6000 kg/m³")
        print("   - Polymers: 900-2000 kg/m³")
        
        rho = self._get_numeric_input("Density (kg/m³)", 100, 25000, None)
        
        # Specific heat
        print("\n3. Specific Heat Capacity")
        print("   Typical values:")
        print("   - Metals: 100-1000 J/kg·K")
        print("   - Semiconductors: 300-1000 J/kg·K")
        print("   - Polymers: 1000-2500 J/kg·K")
        
        cp = self._get_numeric_input("Specific heat (J/kg·K)", 50, 5000, None)
        
        # Category
        print("\n4. Material Category")
        categories = ['metal', 'semiconductor', 'polymer', 'interface_material', 'composite', 'other']
        for i, cat in enumerate(categories, 1):
            print(f"   {i}. {cat}")
        
        cat_choice = int(input("Select category (1-6): "))
        category = categories[cat_choice - 1]
        
        # Contact resistance (optional)
        contact_res = None
        if category == 'interface_material':
            print("\n5. Contact Resistance (optional)")
            print("   Typical values: 1e-6 to 1e-4 m²·K/W")
            if input("Add contact resistance? (y/n) [y]: ").strip().lower() != 'n':
                contact_res = self._get_numeric_input("Contact resistance (m²·K/W)", 1e-7, 1e-3, 1e-5)
        
        # Build material data dictionary
        material_data = {
            "name": name,
            "thermal_conductivity": {
                "value": k,
                "unit": "W/m·K",
                "temperature_ref": 300,
                "temperature_coefficient": k_temp_coeff
            },
            "density": {
                "value": rho,
                "unit": "kg/m³"
            },
            "specific_heat": {
                "value": cp,
                "unit": "J/kg·K"
            },
            "category": category
        }
        
        if contact_res is not None:
            material_data["contact_resistance"] = {
                "value": contact_res,
                "unit": "m²·K/W"
            }
        
        # Show summary
        print("\n" + "-"*50)
        print("MATERIAL SUMMARY:")
        print(f"Name: {name}")
        print(f"Key: {key}")
        print(f"Thermal conductivity: {k} W/m·K")
        if k_temp_coeff > 0:
            print(f"  Temperature coefficient: {k_temp_coeff}")
        print(f"Density: {rho} kg/m³")
        print(f"Specific heat: {cp} J/kg·K")
        print(f"Category: {category}")
        if contact_res:
            print(f"Contact resistance: {contact_res} m²·K/W")
        
        return key, material_data
    
    def show_material_properties(self, material_key: str):
        """Display material properties in detail"""
        try:
            mat = self.data.get_material(material_key)
            print(f"\n{mat.name} Properties:")
            print(f"  Thermal conductivity: {mat.k} W/m·K")
            if mat.k_temp_coeff > 0:
                print(f"    Temperature dependent: k(T) = {mat.k} × (300/T)^{mat.k_temp_coeff}")
            print(f"  Density: {mat.rho} kg/m³")
            print(f"  Specific heat: {mat.cp} J/kg·K")
            print(f"  Category: {mat.category}")
            if mat.contact_resistance:
                print(f"  Contact resistance: {mat.contact_resistance} m²·K/W")
            
            # Calculate thermal diffusivity
            alpha = mat.k / (mat.rho * mat.cp)
            print(f"  Thermal diffusivity: {alpha:.2e} m²/s")
            
        except ValueError:
            # Check custom materials
            if material_key in self.custom_materials:
                mat_data = self.custom_materials[material_key]
                print(f"\n{mat_data['name']} Properties (Custom):")
                k = mat_data['thermal_conductivity']['value']
                rho = mat_data['density']['value']
                cp = mat_data['specific_heat']['value']
                print(f"  Thermal conductivity: {k} W/m·K")
                print(f"  Density: {rho} kg/m³")
                print(f"  Specific heat: {cp} J/kg·K")
                print(f"  Category: {mat_data['category']}")
            else:
                print(f"❌ Material '{material_key}' not found")
    
    def compare_materials(self, material_keys: List[str]):
        """Compare multiple materials side by side"""
        print("\n" + "="*80)
        print("MATERIAL COMPARISON")
        print("="*80)
        
        # Headers
        headers = ['Property', 'Unit'] + [key[:15] for key in material_keys]
        col_widths = [20, 10] + [15] * len(material_keys)
        
        # Print header row
        header_row = ""
        for header, width in zip(headers, col_widths):
            header_row += f"{header:<{width}}"
        print(header_row)
        print("-" * sum(col_widths))
        
        # Get material properties
        materials = []
        for key in material_keys:
            try:
                mat = self.data.get_material(key)
                materials.append({
                    'name': mat.name,
                    'k': mat.k,
                    'rho': mat.rho,
                    'cp': mat.cp,
                    'alpha': mat.k / (mat.rho * mat.cp),
                    'category': mat.category
                })
            except:
                materials.append(None)
        
        # Print properties
        properties = [
            ('Name', '', lambda m: m['name'] if m else 'N/A'),
            ('Thermal Cond.', 'W/m·K', lambda m: f"{m['k']:.1f}" if m else 'N/A'),
            ('Density', 'kg/m³', lambda m: f"{m['rho']:.0f}" if m else 'N/A'),
            ('Specific Heat', 'J/kg·K', lambda m: f"{m['cp']:.0f}" if m else 'N/A'),
            ('Thermal Diff.', 'm²/s', lambda m: f"{m['alpha']:.2e}" if m else 'N/A'),
            ('Category', '', lambda m: m['category'] if m else 'N/A')
        ]
        
        for prop_name, unit, getter in properties:
            row = f"{prop_name:<20}{unit:<10}"
            for mat in materials:
                row += f"{getter(mat):<15}"
            print(row)
    
    def _get_numeric_input(self, prompt: str, min_val: float, max_val: float, default: Optional[float]) -> float:
        """Get numeric input with validation"""
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} [{default}]: ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                value = float(user_input)
                
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"❌ Value must be between {min_val} and {max_val}")
            except ValueError:
                print("❌ Please enter a valid number")


class LayerStackEditor:
    """Dedicated class for layer stack editing operations"""
    
    def __init__(self, data_puller: DataPuller, initial_layers: List[LayerInfo], material_editor: MaterialEditor):
        self.data = data_puller
        self.layers = copy.deepcopy(initial_layers)
        self.material_editor = material_editor
        self.modified = False
        self.original_layers = copy.deepcopy(initial_layers)
    
    def edit_layer_thickness(self, layer_index: int):
        """Edit layer thickness with unit handling"""
        layer = self.layers[layer_index]
        current_thickness_mm = layer.thickness * 1000
        
        print(f"\nEditing thickness for layer: {layer.name}")
        print(f"Current thickness: {current_thickness_mm:.3f} mm ({layer.thickness:.6f} m)")
        
        # Unit selection
        print("\nEnter new thickness in:")
        print("1. Millimeters (mm)")
        print("2. Micrometers (μm)")
        print("3. Meters (m)")
        
        unit_choice = self._get_choice("Select unit", 3)
        
        if unit_choice == 1:
            # Millimeters
            new_thickness_mm = self._get_numeric_input(
                "New thickness (mm)", 0.001, 100, current_thickness_mm
            )
            new_thickness = new_thickness_mm / 1000  # Convert to meters
        elif unit_choice == 2:
            # Micrometers
            current_thickness_um = current_thickness_mm * 1000
            new_thickness_um = self._get_numeric_input(
                "New thickness (μm)", 1, 100000, current_thickness_um
            )
            new_thickness = new_thickness_um / 1e6  # Convert to meters
        else:
            # Meters
            new_thickness = self._get_numeric_input(
                "New thickness (m)", 0.000001, 0.1, layer.thickness
            )
        
        if abs(new_thickness - layer.thickness) > 1e-9:
            self.layers[layer_index].thickness = new_thickness
            self.modified = True
            print(f"✅ Thickness updated: {new_thickness*1000:.3f} mm")
            
            # Show thermal resistance impact
            try:
                mat = self.data.get_material(layer.material_key)
                old_R = layer.thickness / (mat.k * layer.area)
                new_R = new_thickness / (mat.k * layer.area)
                print(f"   Thermal resistance change: {old_R:.6f} → {new_R:.6f} K/W")
                print(f"   Change: {(new_R/old_R - 1)*100:+.1f}%")
            except:
                pass
    
    def edit_layer_area(self, layer_index: int):
        """Edit layer area with unit handling"""
        layer = self.layers[layer_index]
        current_area_mm2 = layer.area * 1e6
        
        print(f"\nEditing area for layer: {layer.name}")
        print(f"Current area: {current_area_mm2:.1f} mm² ({layer.area:.6f} m²)")
        
        # Show equivalent square dimensions
        side_mm = (current_area_mm2 ** 0.5)
        print(f"Equivalent square: {side_mm:.1f} × {side_mm:.1f} mm")
        
        print("\nEnter new area:")
        print("1. By area in mm²")
        print("2. By area in cm²")
        print("3. By area in m²")
        print("4. By square dimensions (mm)")
        
        method_choice = self._get_choice("Select method", 4)
        
        if method_choice == 1:
            # Square millimeters
            new_area_mm2 = self._get_numeric_input(
                "New area (mm²)", 1, 10000, current_area_mm2
            )
            new_area = new_area_mm2 / 1e6
        elif method_choice == 2:
            # Square centimeters
            current_area_cm2 = current_area_mm2 / 100
            new_area_cm2 = self._get_numeric_input(
                "New area (cm²)", 0.01, 100, current_area_cm2
            )
            new_area = new_area_cm2 / 1e4
        elif method_choice == 3:
            # Square meters
            new_area = self._get_numeric_input(
                "New area (m²)", 0.000001, 0.01, layer.area
            )
        else:
            # By square dimensions
            print("\nEnter square side length:")
            side_mm = self._get_numeric_input(
                "Side length (mm)", 1, 100, side_mm
            )
            new_area = (side_mm / 1000) ** 2
        
        if abs(new_area - layer.area) > 1e-9:
            self.layers[layer_index].area = new_area
            self.modified = True
            print(f"✅ Area updated: {new_area*1e6:.1f} mm²")
            
            # Check for area mismatch with adjacent layers
            if layer_index > 0 and new_area < self.layers[layer_index-1].area * 0.9:
                print("⚠️  Warning: Significant area reduction from previous layer")
                print("   This will create spreading resistance")
            if layer_index < len(self.layers)-1 and new_area > self.layers[layer_index+1].area * 1.1:
                print("⚠️  Warning: Area larger than next layer")
                print("   Consider adjusting adjacent layers")
    
    def substitute_material(self, layer_index: int):
        """Substitute layer material"""
        layer = self.layers[layer_index]
        
        print(f"\nSubstituting material for layer: {layer.name}")
        print(f"Current material: {layer.material_key}")
        
        # Show current material properties
        self.material_editor.show_material_properties(layer.material_key)
        
        print("\nSelect new material:")
        print("1. Choose from existing materials")
        print("2. Create custom material")
        print("3. Compare materials")
        print("4. Cancel")
        
        choice = self._get_choice("Select option", 4)
        
        if choice == 1:
            # List available materials by category
            materials = self.data.list_available_materials()
            
            # Get material categories
            categories = {}
            for mat_key in materials:
                try:
                    mat = self.data.get_material(mat_key)
                    if mat.category not in categories:
                        categories[mat.category] = []
                    categories[mat.category].append((mat_key, mat))
                except:
                    pass
            
            # Display by category
            all_mats = []
            i = 1
            for category, mats in sorted(categories.items()):
                print(f"\n{category.upper()}:")
                for mat_key, mat in sorted(mats, key=lambda x: x[1].name):
                    print(f"{i:2d}. {mat.name:<30} (k={mat.k:>6.1f} W/m·K)")
                    all_mats.append(mat_key)
                    i += 1
            
            mat_choice = self._get_choice("Select material", len(all_mats))
            new_material_key = all_mats[mat_choice - 1]
            
        elif choice == 2:
            # Create custom material
            new_material_key, material_data = self.material_editor.create_material()
            self.material_editor.custom_materials[new_material_key] = material_data
            
        elif choice == 3:
            # Compare materials
            print("\nEnter material keys to compare (comma-separated):")
            mat_keys = input("Material keys: ").strip().split(',')
            mat_keys = [k.strip() for k in mat_keys]
            self.material_editor.compare_materials(mat_keys)
            return self.substitute_material(layer_index)  # Retry
        else:
            return  # Cancel
        
        # Update material
        if new_material_key != layer.material_key:
            old_key = layer.material_key
            self.layers[layer_index].material_key = new_material_key
            self.modified = True
            print(f"✅ Material updated: {old_key} → {new_material_key}")
            
            # Show thermal impact
            try:
                old_mat = self.data.get_material(old_key)
                new_mat = self.data.get_material(new_material_key)
                old_R = layer.thickness / (old_mat.k * layer.area)
                new_R = layer.thickness / (new_mat.k * layer.area)
                print(f"   Thermal resistance change: {old_R:.6f} → {new_R:.6f} K/W")
                print(f"   Change: {(new_R/old_R - 1)*100:+.1f}%")
            except:
                pass
    
    def add_layer(self, position: Optional[int] = None):
        """Add new layer to stack"""
        print("\n" + "="*50)
        print("ADD NEW LAYER")
        print("="*50)
        
        # Get layer name
        name = input("Layer name: ").strip()
        if not name:
            print("❌ Layer name required")
            return
        
        # Get thickness
        print("\nLayer thickness:")
        print("1. Enter in millimeters")
        print("2. Enter in micrometers")
        
        unit_choice = self._get_choice("Select unit", 2)
        
        if unit_choice == 1:
            thickness_mm = self._get_numeric_input("Thickness (mm)", 0.001, 10, None)
            thickness = thickness_mm / 1000
        else:
            thickness_um = self._get_numeric_input("Thickness (μm)", 1, 10000, None)
            thickness = thickness_um / 1e6
        
        # Get area (default to previous layer's area if available)
        default_area = self.layers[-1].area if self.layers else 0.0004
        print(f"\nLayer area (default: {default_area*1e6:.1f} mm²):")
        area_mm2 = self._get_numeric_input("Area (mm²)", 1, 10000, default_area*1e6)
        area = area_mm2 / 1e6
        
        # Select material
        print("\nSelect material:")
        materials = self.data.list_available_materials()
        for i, mat_key in enumerate(materials[:10], 1):
            try:
                mat = self.data.get_material(mat_key)
                print(f"{i:2d}. {mat.name}")
            except:
                pass
        print("11. Show all materials")
        print("12. Create custom material")
        
        mat_choice = self._get_choice("Select option", 12)
        
        if mat_choice <= 10:
            material_key = materials[mat_choice - 1]
        elif mat_choice == 11:
            # Show all materials
            for i, mat_key in enumerate(materials, 1):
                try:
                    mat = self.data.get_material(mat_key)
                    print(f"{i:2d}. {mat.name}")
                except:
                    pass
            mat_idx = self._get_choice("Select material", len(materials))
            material_key = materials[mat_idx - 1]
        else:
            # Create custom material
            material_key, material_data = self.material_editor.create_material()
            self.material_editor.custom_materials[material_key] = material_data
        
        # Create new layer
        new_layer = LayerInfo(
            name=name,
            thickness=thickness,
            area=area,
            material_key=material_key,
            notes=input("Notes (optional): ").strip()
        )
        
        # Get position
        if position is None:
            print(f"\nWhere to insert layer? (1-{len(self.layers)+1})")
            for i in range(len(self.layers) + 1):
                if i == 0:
                    print(f"1. At the beginning (before {self.layers[0].name})")
                elif i == len(self.layers):
                    print(f"{i+1}. At the end (after {self.layers[-1].name})")
                else:
                    print(f"{i+1}. Between {self.layers[i-1].name} and {self.layers[i].name}")
            
            position = self._get_choice("Position", len(self.layers) + 1) - 1
        
        # Insert layer
        self.layers.insert(position, new_layer)
        self.modified = True
        print(f"✅ Layer '{name}' added at position {position + 1}")
    
    def remove_layer(self, layer_index: int):
        """Remove layer from stack"""
        if len(self.layers) <= 1:
            print("❌ Cannot remove last layer")
            return
        
        layer = self.layers[layer_index]
        confirm = input(f"Remove layer '{layer.name}'? (y/n): ").strip().lower()
        
        if confirm == 'y':
            removed = self.layers.pop(layer_index)
            self.modified = True
            print(f"✅ Layer '{removed.name}' removed")
    
    def reorder_layers(self):
        """Reorder layers in stack"""
        if len(self.layers) < 2:
            print("❌ Need at least 2 layers to reorder")
            return
        
        print("\nCurrent layer order:")
        for i, layer in enumerate(self.layers, 1):
            print(f"{i}. {layer.name}")
        
        # Get layer to move
        from_idx = self._get_choice("Select layer to move", len(self.layers)) - 1
        
        # Get new position
        print(f"\nMove '{self.layers[from_idx].name}' to position:")
        for i in range(len(self.layers)):
            if i == from_idx:
                continue
            if i == 0:
                print(f"1. To the beginning")
            else:
                print(f"{i+1}. After {self.layers[i-1].name}")
        
        to_idx = self._get_choice("New position", len(self.layers)) - 1
        
        # Reorder
        layer = self.layers.pop(from_idx)
        if to_idx > from_idx:
            to_idx -= 1
        self.layers.insert(to_idx, layer)
        self.modified = True
        
        print(f"✅ Layer reordered")
        print("\nNew order:")
        for i, layer in enumerate(self.layers, 1):
            print(f"{i}. {layer.name}")
    
    def get_stack_summary(self) -> str:
        """Get formatted stack summary"""
        summary = []
        total_thickness = 0
        total_resistance_estimate = 0
        
        summary.append("\n" + "="*80)
        summary.append("LAYER STACK SUMMARY")
        summary.append("="*80)
        summary.append(f"{'Layer':<15} {'Thickness':<12} {'Area':<12} {'Material':<20} {'k (W/m·K)':<10}")
        summary.append("-"*80)
        
        for i, layer in enumerate(self.layers):
            thickness_mm = layer.thickness * 1000
            area_mm2 = layer.area * 1e6
            total_thickness += thickness_mm
            
            try:
                mat = self.data.get_material(layer.material_key)
                k = mat.k
                R_est = layer.thickness / (k * layer.area)
                total_resistance_estimate += R_est
            except:
                k = "?"
                R_est = 0
            
            summary.append(
                f"{layer.name:<15} {thickness_mm:<12.3f} {area_mm2:<12.1f} "
                f"{layer.material_key:<20} {k:<10}"
            )
        
        summary.append("-"*80)
        summary.append(f"Total thickness: {total_thickness:.3f} mm")
        if total_resistance_estimate > 0:
            summary.append(f"Estimated total conduction resistance: {total_resistance_estimate:.6f} K/W")
        
        # Check for issues
        issues = []
        for i in range(len(self.layers) - 1):
            if self.layers[i].area < self.layers[i+1].area * 0.5:
                issues.append(f"⚠️  Large area increase from {self.layers[i].name} to {self.layers[i+1].name}")
            elif self.layers[i].area > self.layers[i+1].area * 2:
                issues.append(f"⚠️  Large area decrease from {self.layers[i].name} to {self.layers[i+1].name}")
        
        if issues:
            summary.append("\nPotential Issues:")
            summary.extend(issues)
        
        return "\n".join(summary)
    
    def show_modifications(self):
        """Show what has been modified from original"""
        if not self.modified:
            print("\nNo modifications made")
            return
        
        print("\n" + "="*60)
        print("MODIFICATIONS FROM ORIGINAL CONFIGURATION")
        print("="*60)
        
        # Compare layer counts
        if len(self.layers) != len(self.original_layers):
            print(f"Layer count: {len(self.original_layers)} → {len(self.layers)}")
        
        # Compare individual layers
        for i, (orig, curr) in enumerate(zip(self.original_layers, self.layers)):
            changes = []
            
            if orig.thickness != curr.thickness:
                changes.append(f"thickness: {orig.thickness*1000:.3f} → {curr.thickness*1000:.3f} mm")
            if orig.area != curr.area:
                changes.append(f"area: {orig.area*1e6:.1f} → {curr.area*1e6:.1f} mm²")
            if orig.material_key != curr.material_key:
                changes.append(f"material: {orig.material_key} → {curr.material_key}")
            
            if changes:
                print(f"\n{curr.name}:")
                for change in changes:
                    print(f"  - {change}")
    
    def _get_choice(self, prompt: str, num_options: int) -> int:
        """Get menu choice with validation"""
        while True:
            try:
                choice = int(input(f"{prompt} (1-{num_options}): "))
                if 1 <= choice <= num_options:
                    return choice
                print(f"❌ Please enter a number between 1 and {num_options}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _get_numeric_input(self, prompt: str, min_val: float, max_val: float, default: Optional[float]) -> float:
        """Get numeric input with validation"""
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} [{default}]: ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                value = float(user_input)
                
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"❌ Value must be between {min_val} and {max_val}")
            except ValueError:
                print("❌ Please enter a valid number")


class AdvancedThermalInputGuide:
    """Advanced thermal input guide with granular customization"""
    
    def __init__(self):
        self.data = DataPuller()
        self.current_config = None
        self.custom_materials = {}
        self.config_modified = False
        self.material_editor = MaterialEditor(self.data)
        self.layer_editor = None
        
        # Load available options
        self.chip_configs = self.data.list_available_configurations()
        self.materials = self.data.list_available_materials()
        self.fluids = self.data.list_available_fluids()
        
        # Extract vapor chamber configs
        self.vc_configs = list(self.data._dimensions_data['vapor_chamber_configs'].keys())
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(f"{title.center(60)}")
        print("="*60)
    
    def get_numeric_input(self, prompt: str, min_val: float = None, 
                         max_val: float = None, default: float = None) -> float:
        """Get numeric input with validation"""
        while True:
            try:
                user_input = input(f"{prompt} [{default}]: " if default else f"{prompt}: ").strip()
                
                if user_input == "" and default is not None:
                    return default
                
                value = float(user_input)
                
                if min_val is not None and value < min_val:
                    print(f"❌ Value must be at least {min_val}")
                    continue
                    
                if max_val is not None and value > max_val:
                    print(f"❌ Value must be at most {max_val}")
                    continue
                    
                return value
                
            except ValueError:
                print("❌ Please enter a valid number")
    
    def get_choice(self, prompt: str, num_options: int) -> int:
        """Get menu choice with validation"""
        while True:
            try:
                choice = int(input(f"\n{prompt}: "))
                if 1 <= choice <= num_options:
                    return choice
                print(f"❌ Please enter a number between 1 and {num_options}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def temperature_to_kelvin(self, temp: float, unit: str) -> float:
        """Convert temperature to Kelvin"""
        if unit.upper() == 'C':
            return temp + 273.15
        elif unit.upper() == 'F':
            return (temp - 32) * 5/9 + 273.15
        else:  # Already in Kelvin
            return temp
    
    def get_temperature_input(self, prompt: str, default_celsius: float = 25) -> float:
        """Get temperature input with unit conversion"""
        print(f"\n{prompt}")
        print("1. Enter in Celsius")
        print("2. Enter in Kelvin")
        print("3. Enter in Fahrenheit")
        
        unit_choice = self.get_choice("Select temperature unit", 3)
        
        if unit_choice == 1:
            temp = self.get_numeric_input(
                f"Temperature (°C)", 
                min_val=-273.15, 
                max_val=200, 
                default=default_celsius
            )
            return self.temperature_to_kelvin(temp, 'C')
        elif unit_choice == 2:
            return self.get_numeric_input(
                f"Temperature (K)", 
                min_val=0, 
                max_val=473.15, 
                default=default_celsius + 273.15
            )
        else:
            temp = self.get_numeric_input(
                f"Temperature (°F)", 
                min_val=-459.67, 
                max_val=392, 
                default=default_celsius * 9/5 + 32
            )
            return self.temperature_to_kelvin(temp, 'F')
    
    def run_interactive_guide(self) -> Optional[Dict[str, Any]]:
        """Main interactive menu system with advanced options"""
        self.clear_screen()
        self.print_header("ADVANCED GPU THERMAL ANALYSIS CONFIGURATION TOOL")
        print("\nGranular customization of thermal analysis parameters")
        print("Modify layer stacks, materials, and cooling configurations")
        
        input("\nPress Enter to begin...")
        
        try:
            # Step 1: Configuration Setup
            config_result = self.configure_base_setup()
            if not config_result:
                return None
            
            # Step 2: Layer Stack Customization
            if config_result['customize_layers']:
                customize_result = self.configure_layer_stack()
                if not customize_result:
                    return None
            
            # Step 3: Power and Environment
            power_watts = self.get_power_input()
            T_ambient = self.get_ambient_temperature()
            
            # Step 4: Cooling Configuration
            cooling_type = self.select_cooling_type()
            cooling_params = self.get_cooling_parameters(cooling_type)
            
            # Build configuration dictionary
            config = {
                'config_key': self.current_config,
                'power_watts': power_watts,
                'T_ambient': T_ambient,
                'cooling_type': cooling_type,
                'cooling_params': cooling_params,
                'custom_layers': self.layer_editor.layers if self.layer_editor else None,
                'custom_materials': self.material_editor.custom_materials
            }
            
            # Step 5: Final Workflow Choice
            return self.final_workflow_menu(config)
            
        except KeyboardInterrupt:
            print("\n\n❌ Configuration cancelled by user")
            return None
    
    def configure_base_setup(self) -> Optional[Dict[str, Any]]:
        """Initial configuration setup"""
        self.clear_screen()
        self.print_header("CONFIGURATION SETUP")
        
        print("\nSelect starting point:")
        print("1. Load existing chip configuration")
        print("2. Create custom configuration from scratch")
        print("3. Import configuration from file")
        
        choice = self.get_choice("Select option", 3)
        
        if choice == 1:
            # Load existing configuration
            self.current_config = self.select_chip_configuration()
            
            print("\nConfiguration loaded. Options:")
            print("1. Use as-is")
            print("2. Customize layer stack")
            print("3. View detailed configuration")
            
            option = self.get_choice("Select option", 3)
            
            if option == 1:
                return {'customize_layers': False}
            elif option == 2:
                return {'customize_layers': True}
            else:
                self.show_detailed_config(self.current_config)
                return self.configure_base_setup()
                
        elif choice == 2:
            # Create from scratch
            print("\nCreating custom configuration...")
            self.current_config = "custom_config"
            
            # Initialize with single layer
            default_layer = LayerInfo(
                name="die",
                thickness=0.000775,
                area=0.0004,
                material_key="silicon",
                notes="Custom die"
            )
            
            self.layer_editor = LayerStackEditor(
                self.data, 
                [default_layer],
                self.material_editor
            )
            
            return {'customize_layers': True}
            
        else:
            # Import from file (placeholder)
            print("\n❌ Import feature not yet implemented")
            return self.configure_base_setup()
    
    def select_chip_configuration(self) -> str:
        """Guide user through chip configuration selection"""
        self.clear_screen()
        self.print_header("CHIP CONFIGURATION SELECTION")
        
        print("\nAvailable configurations:")
        for i, config_key in enumerate(self.chip_configs, 1):
            config_data = self.data._dimensions_data['chip_configurations'][config_key]
            num_layers = len(config_data['layers'])
            print(f"{i}. {config_data['name']} ({num_layers} layers) - {config_key}")
        
        print(f"\n{len(self.chip_configs) + 1}. View all configuration details")
        
        choice = self.get_choice("Select configuration", len(self.chip_configs) + 1)
        
        if choice == len(self.chip_configs) + 1:
            self.show_all_configurations()
            return self.select_chip_configuration()
        
        return self.chip_configs[choice - 1]
    
    def show_detailed_config(self, config_key: str):
        """Show detailed configuration information"""
        config_data = self.data._dimensions_data['chip_configurations'][config_key]
        layers = self.data.get_chip_configuration(config_key)
        
        print(f"\n{'='*80}")
        print(f"{config_data['name']} - Detailed View")
        print(f"{'='*80}")
        
        total_thickness = 0
        total_resistance = 0
        
        print(f"\n{'Layer':<15} {'Thickness':<15} {'Area':<15} {'Material':<20} {'k (W/m·K)':<10}")
        print("-"*80)
        
        for layer in layers:
            thickness_mm = layer.thickness * 1000
            area_mm2 = layer.area * 1e6
            total_thickness += thickness_mm
            
            try:
                mat = self.data.get_material(layer.material_key)
                k = f"{mat.k:.1f}"
                R = layer.thickness / (mat.k * layer.area)
                total_resistance += R
            except:
                k = "N/A"
                R = 0
            
            print(f"{layer.name:<15} {thickness_mm:<15.3f} {area_mm2:<15.1f} {layer.material_key:<20} {k:<10}")
            
            if layer.notes:
                print(f"  Notes: {layer.notes}")
        
        print("-"*80)
        print(f"Total thickness: {total_thickness:.3f} mm")
        print(f"Estimated conduction resistance: {total_resistance:.6f} K/W")
        
        input("\nPress Enter to continue...")
    
    def show_all_configurations(self):
        """Show all configuration details"""
        for config_key in self.chip_configs:
            self.show_detailed_config(config_key)
    
    def configure_layer_stack(self) -> bool:
        """Advanced layer stack configuration"""
        self.clear_screen()
        self.print_header("LAYER STACK CUSTOMIZATION")
        
        # Initialize layer editor if not already done
        if not self.layer_editor:
            layers = self.data.get_chip_configuration(self.current_config)
            self.layer_editor = LayerStackEditor(self.data, layers, self.material_editor)
        
        while True:
            # Show current stack
            print(self.layer_editor.get_stack_summary())
            
            print("\n" + "="*60)
            print("LAYER STACK EDITING OPTIONS")
            print("="*60)
            print("1. Edit layer properties (thickness/area)")
            print("2. Substitute layer material")
            print("3. Add new layer")
            print("4. Remove layer")
            print("5. Reorder layers")
            print("6. Show modifications from original")
            print("7. Reset to original configuration")
            print("8. Done editing")
            
            choice = self.get_choice("Select option", 8)
            
            if choice == 1:
                # Edit layer properties
                self.edit_layer_properties()
            
            elif choice == 2:
                # Substitute material
                layer_idx = self.select_layer_for_editing()
                if layer_idx is not None:
                    self.layer_editor.substitute_material(layer_idx)
            
            elif choice == 3:
                # Add layer
                self.layer_editor.add_layer()
            
            elif choice == 4:
                # Remove layer
                layer_idx = self.select_layer_for_editing()
                if layer_idx is not None:
                    self.layer_editor.remove_layer(layer_idx)
            
            elif choice == 5:
                # Reorder layers
                self.layer_editor.reorder_layers()
            
            elif choice == 6:
                # Show modifications
                self.layer_editor.show_modifications()
                input("\nPress Enter to continue...")
            
            elif choice == 7:
                # Reset to original
                confirm = input("Reset all changes? (y/n): ").strip().lower()
                if confirm == 'y':
                    layers = self.data.get_chip_configuration(self.current_config)
                    self.layer_editor = LayerStackEditor(self.data, layers, self.material_editor)
                    print("✅ Reset to original configuration")
            
            else:
                # Done editing
                if self.layer_editor.modified:
                    self.config_modified = True
                return True
    
    def edit_layer_properties(self):
        """Edit individual layer properties submenu"""
        layer_idx = self.select_layer_for_editing()
        if layer_idx is None:
            return
        
        while True:
            layer = self.layer_editor.layers[layer_idx]
            print(f"\nEditing: {layer.name}")
            print(f"1. Edit thickness (current: {layer.thickness*1000:.3f} mm)")
            print(f"2. Edit area (current: {layer.area*1e6:.1f} mm²)")
            print(f"3. Edit both")
            print(f"4. Back")
            
            choice = self.get_choice("Select property", 4)
            
            if choice == 1:
                self.layer_editor.edit_layer_thickness(layer_idx)
            elif choice == 2:
                self.layer_editor.edit_layer_area(layer_idx)
            elif choice == 3:
                self.layer_editor.edit_layer_thickness(layer_idx)
                self.layer_editor.edit_layer_area(layer_idx)
            else:
                break
    
    def select_layer_for_editing(self) -> Optional[int]:
        """Select a layer for editing"""
        print("\nSelect layer:")
        for i, layer in enumerate(self.layer_editor.layers, 1):
            print(f"{i}. {layer.name}")
        print(f"{len(self.layer_editor.layers) + 1}. Cancel")
        
        choice = self.get_choice("Select layer", len(self.layer_editor.layers) + 1)
        
        if choice == len(self.layer_editor.layers) + 1:
            return None
        
        return choice - 1
    
    def get_power_input(self) -> float:
        """Get power dissipation input"""
        self.clear_screen()
        self.print_header("POWER DISSIPATION")
        
        print("\nTypical GPU power ranges:")
        print("- Entry-level GPU: 75-150W")
        print("- Mid-range GPU: 150-250W")
        print("- High-end GPU: 250-350W")
        print("- Extreme GPU: 350-500W+")
        
        return self.get_numeric_input(
            "\nEnter total power dissipation (W)",
            min_val=10,
            max_val=3000,
            default=150
        )
    
    def get_ambient_temperature(self) -> float:
        """Get ambient temperature input"""
        self.clear_screen()
        self.print_header("AMBIENT TEMPERATURE")
        
        print("\nTypical ambient temperatures:")
        print("- Cool room: 18-20°C")
        print("- Normal room: 22-25°C")
        print("- Warm room: 28-30°C")
        print("- Hot environment: 35-40°C")
        
        return self.get_temperature_input("Enter ambient temperature", default_celsius=25)
    
    def select_cooling_type(self) -> CoolingType:
        """Guide user through cooling type selection"""
        self.clear_screen()
        self.print_header("COOLING TYPE SELECTION")
        
        cooling_descriptions = {
            CoolingType.AIR: "Traditional air cooling with heatsink and fan",
            CoolingType.LIQUID: "Liquid cooling with cold plate and coolant",
            CoolingType.EVAPORATIVE: "Two-phase cooling using boiling/evaporation",
            CoolingType.VAPOR_CHAMBER: "Advanced heat spreading with vapor chambers",
            CoolingType.HYBRID: "Combination of multiple cooling methods"
        }
        
        print("\nAvailable cooling types:")
        cooling_types = list(CoolingType)
        for i, cooling_type in enumerate(cooling_types, 1):
            print(f"{i}. {cooling_type.value:<15} - {cooling_descriptions[cooling_type]}")
        
        print(f"\n{len(cooling_types) + 1}. Learn more about cooling types")
        
        choice = self.get_choice("Select cooling type", len(cooling_types) + 1)
        
        if choice == len(cooling_types) + 1:
            self.show_cooling_type_details()
            return self.select_cooling_type()
        
        selected_type = cooling_types[choice - 1]
        print(f"\n✅ Selected: {selected_type.value}")
        
        return selected_type
    
    def show_cooling_type_details(self):
        """Show detailed information about cooling types"""
        self.clear_screen()
        self.print_header("COOLING TYPE DETAILS")
        
        details = {
            CoolingType.AIR: """
Air Cooling:
- Uses heatsink with fins to increase surface area
- Fan provides forced convection
- Typical HTC: 25-250 W/m²·K
- Simple, reliable, but limited cooling capacity
- Best for: Low to medium power GPUs (<200W)
            """,
            CoolingType.LIQUID: """
Liquid Cooling:
- Uses cold plate with liquid coolant (water/glycol)
- Much higher heat transfer than air
- Typical HTC: 1000-10000 W/m²·K
- Requires pump, radiator, and tubing
- Best for: High-performance GPUs (200-400W)
            """,
            CoolingType.EVAPORATIVE: """
Evaporative Cooling:
- Uses phase change (liquid to vapor) for heat removal
- Extremely high heat transfer coefficients
- Nucleate boiling HTC: 10000-100000 W/m²·K
- Limited by critical heat flux
- Best for: Extreme cooling applications
            """,
            CoolingType.VAPOR_CHAMBER: """
Vapor Chamber Cooling:
- Sealed chamber with working fluid
- Evaporation at hot spot, condensation at cold regions
- Very high effective thermal conductivity
- Excellent heat spreading capability
- Best for: High heat flux with spreading needs
            """,
            CoolingType.HYBRID: """
Hybrid Cooling:
- Combines multiple cooling methods
- Example: Vapor chamber + liquid cooling
- Maximizes cooling performance
- Complex but highly effective
- Best for: Extreme performance applications
            """
        }
        
        for cooling_type, description in details.items():
            print(f"\n{cooling_type.value.upper()}:{description}")
        
        input("\nPress Enter to continue...")
    
    def get_cooling_parameters(self, cooling_type: CoolingType) -> Dict[str, Any]:
        """Get parameters specific to cooling type"""
        if cooling_type == CoolingType.AIR:
            return self.get_air_cooling_params()
        elif cooling_type == CoolingType.LIQUID:
            return self.get_liquid_cooling_params()
        elif cooling_type == CoolingType.EVAPORATIVE:
            return self.get_evaporative_cooling_params()
        elif cooling_type == CoolingType.VAPOR_CHAMBER:
            return self.get_vapor_chamber_params()
        elif cooling_type == CoolingType.HYBRID:
            return self.get_hybrid_cooling_params()
        else:
            raise ValueError(f"Unknown cooling type: {cooling_type}")
    
    def get_air_cooling_params(self) -> Dict[str, float]:
        """Get parameters for air cooling"""
        self.clear_screen()
        self.print_header("AIR COOLING PARAMETERS")
        
        # Heat transfer coefficient
        print("\n1. Heat Transfer Coefficient (h_air)")
        print("This depends on airflow velocity and heatsink design")
        
        # Get ranges from JSON
        natural_range = self.data.get_cooling_htc_range('air', 'natural_convection')['h_range']
        forced_range = self.data.get_cooling_htc_range('air', 'forced_convection')['h_range']
        
        print(f"\nTypical ranges:")
        print(f"- Natural convection: {natural_range[0]}-{natural_range[1]} W/m²·K")
        print(f"- Forced convection: {forced_range[0]}-{forced_range[1]} W/m²·K")
        
        print("\nSelect option:")
        print("1. Low airflow (quiet)")
        print("2. Medium airflow (balanced)")
        print("3. High airflow (performance)")
        print("4. Custom value")
        
        htc_choice = self.get_choice("Select option", 4)
        
        if htc_choice == 1:
            h_air = 30
        elif htc_choice == 2:
            h_air = 50
        elif htc_choice == 3:
            h_air = 100
        else:
            h_air = self.get_numeric_input(
                "Enter h_air (W/m²·K)",
                min_val=5,
                max_val=250,
                default=50
            )
        
        # Heatsink area
        print("\n2. Heatsink Area")
        print("Total surface area including fins")
        
        # Get heatsink options from JSON
        air_heatsinks = self.data._dimensions_data['heatsink_dimensions']['air_cooled']
        
        print("\nSelect heatsink size:")
        options = []
        for i, (size, dims) in enumerate(air_heatsinks.items(), 1):
            fin_area = dims['fin_area']
            print(f"{i}. {size.capitalize()} heatsink - {fin_area} m² ({fin_area*10000:.0f} cm²)")
            options.append((size, fin_area))
        print(f"{len(options) + 1}. Custom area")
        
        area_choice = self.get_choice("Select option", len(options) + 1)
        
        if area_choice <= len(options):
            A_sink = options[area_choice - 1][1]
        else:
            A_sink = self.get_numeric_input(
                "Enter heatsink area (m²)",
                min_val=0.001,
                max_val=1.0,
                default=0.1
            )
        
        params = {
            'h_air': h_air,
            'A_sink': A_sink
        }
        
        print(f"\n✅ Air cooling parameters:")
        print(f"   h_air = {h_air} W/m²·K")
        print(f"   A_sink = {A_sink} m² ({A_sink*10000:.0f} cm²)")
        
        return params
    
    def get_liquid_cooling_params(self) -> Dict[str, float]:
        """Get parameters for liquid cooling"""
        self.clear_screen()
        self.print_header("LIQUID COOLING PARAMETERS")
        
        # Heat transfer coefficient
        print("\n1. Heat Transfer Coefficient (h_liquid)")
        print("Depends on flow rate and cold plate design")
        
        # Get ranges from JSON
        water_range = self.data.get_cooling_htc_range('liquid', 'water_cooling')['h_range']
        micro_range = self.data.get_cooling_htc_range('liquid', 'microchannel')['h_range']
        
        print(f"\nTypical ranges:")
        print(f"- Standard water block: {water_range[0]}-{water_range[1]} W/m²·K")
        print(f"- Microchannel cooler: {micro_range[0]}-{micro_range[1]} W/m²·K")
        
        print("\nSelect option:")
        print("1. Low flow rate")
        print("2. Medium flow rate")
        print("3. High flow rate")
        print("4. Microchannel cooler")
        print("5. Custom value")
        
        htc_choice = self.get_choice("Select option", 5)
        
        if htc_choice == 1:
            h_liquid = 2000
        elif htc_choice == 2:
            h_liquid = 5000
        elif htc_choice == 3:
            h_liquid = 8000
        elif htc_choice == 4:
            h_liquid = 20000
        else:
            h_liquid = self.get_numeric_input(
                "Enter h_liquid (W/m²·K)",
                min_val=1000,
                max_val=100000,
                default=5000
            )
        
        # Cold plate area
        print("\n2. Cold Plate Area")
        
        # Get cold plate options from JSON
        liquid_coolers = self.data._dimensions_data['heatsink_dimensions']['liquid_cooled']
        
        print("\nSelect cold plate type:")
        options = []
        for i, (type_name, dims) in enumerate(liquid_coolers.items(), 1):
            area = dims['base_area']
            print(f"{i}. {type_name.replace('_', ' ').title()} - {area} m² ({area*10000:.1f} cm²)")
            options.append((type_name, area))
        print(f"{len(options) + 1}. Custom area")
        
        area_choice = self.get_choice("Select option", len(options) + 1)
        
        if area_choice <= len(options):
            A_cold = options[area_choice - 1][1]
        else:
            A_cold = self.get_numeric_input(
                "Enter cold plate area (m²)",
                min_val=0.0001,
                max_val=0.01,
                default=0.0016
            )
        
        # Coolant temperature
        print("\n3. Coolant Temperature")
        print("Temperature of the liquid entering the cold plate")
        print("\nTypical coolant temperatures:")
        print("- Room temperature coolant: 25°C")
        print("- Chilled coolant: 15°C")
        print("- Cold coolant: 10°C")
        
        T_coolant = self.get_temperature_input("Enter coolant temperature", default_celsius=15)
        
        params = {
            'h_liquid': h_liquid,
            'A_cold': A_cold,
            'T_coolant': T_coolant
        }
        
        print(f"\n✅ Liquid cooling parameters:")
        print(f"   h_liquid = {h_liquid} W/m²·K")
        print(f"   A_cold = {A_cold} m² ({A_cold*10000:.1f} cm²)")
        print(f"   T_coolant = {T_coolant:.1f} K ({T_coolant-273.15:.1f}°C)")
        
        return params
    
    def get_evaporative_cooling_params(self) -> Dict[str, Any]:
        """Get parameters for evaporative cooling"""
        self.clear_screen()
        self.print_header("EVAPORATIVE COOLING PARAMETERS")
        
        print("\nEvaporative cooling uses phase change for heat removal")
        print("Supports both traditional fluids and refrigerants")
        
        # Check if REFPROP is available
        try:
            from main.utility.refprop_interface import REFPROP_AVAILABLE, RefpropInterface
        except ImportError:
            REFPROP_AVAILABLE = False
            RefpropInterface = None
        
        print("\nSelect fluid type:")
        print("1. Traditional fluids (water, dielectric)")
        if REFPROP_AVAILABLE:
            print("2. Refrigerants (R134a, R1234yf, etc.) - REFPROP")
        else:
            print("2. Refrigerants - ⚠️  REFPROP not available")
        
        fluid_type_choice = self.get_choice("Select option", 2)
        
        if fluid_type_choice == 1:
            # Traditional fluids
            print("\nAvailable fluids:")
            for i, fluid_key in enumerate(self.fluids, 1):
                fluid_data = self.data._materials_data['fluids'][fluid_key]
                T_sat = fluid_data['properties']['saturation_temperature']
                print(f"{i}. {fluid_data['name']} - Boiling point: {T_sat:.1f}K ({T_sat-273.15:.1f}°C)")
            
            fluid_choice = self.get_choice("Select working fluid", len(self.fluids))
            selected_fluid = self.fluids[fluid_choice - 1]
            
            # Show fluid properties
            fluid_props = self.data.get_fluid_properties(selected_fluid)
            print(f"\n✅ Selected fluid: {selected_fluid}")
            print(f"   Saturation temperature: {fluid_props.T_sat:.1f}K ({fluid_props.T_sat-273.15:.1f}°C)")
            print(f"   Latent heat: {fluid_props.h_fg/1e6:.2f} MJ/kg")
            print(f"   Surface tension: {fluid_props.sigma:.4f} N/m")
            
            return {'fluid': selected_fluid, 'use_refprop': False}
            
        else:
            # Refrigerants
            if not REFPROP_AVAILABLE:
                print("\n❌ REFPROP is not available")
                print("To use refrigerants, please:")
                print("1. Install NIST REFPROP software")
                print("2. pip install ctREFPROP")
                print("3. Set RPPREFIX environment variable")
                print("\nFalling back to traditional fluids...")
                return self.get_evaporative_cooling_params()
            
            # List available refrigerants
            print("\nAvailable refrigerants:")
            refrigerants = [
                ('r134a', 'R134a - Common in chillers'),
                ('r1234yf', 'R1234yf - Low GWP alternative'),
                ('r1234ze', 'R1234ze(E) - Low GWP, data center'),
                ('r245fa', 'R245fa - Low pressure, electronics cooling'),
                ('r410a', 'R410a - High pressure blend'),
                ('r32', 'R32 - Moderate pressure, efficient'),
                ('r744', 'CO2 (R744) - Natural, high pressure'),
                ('r717', 'Ammonia (R717) - Industrial'),
                ('water', 'Water - Via REFPROP')
            ]
            
            for i, (key, desc) in enumerate(refrigerants, 1):
                print(f"{i}. {desc}")
            
            ref_choice = self.get_choice("Select refrigerant", len(refrigerants))
            selected_refrigerant = refrigerants[ref_choice - 1][0]
            
            # Get system pressure
            print(f"\n✅ Selected refrigerant: {selected_refrigerant}")
            print("\nSystem pressure determines saturation temperature")
            
            # Show typical pressure ranges
            pressure_ranges = {
                'r134a': (1, 10, 5),
                'r1234yf': (1, 10, 5),
                'r1234ze': (1, 10, 5),
                'r245fa': (0.5, 5, 2),
                'r410a': (5, 30, 15),
                'r32': (5, 25, 12),
                'r744': (20, 70, 35),
                'r717': (1, 15, 8),
                'water': (0.01, 2, 0.1)
            }
            
            if selected_refrigerant in pressure_ranges:
                min_p, max_p, default_p = pressure_ranges[selected_refrigerant]
                print(f"Typical range: {min_p}-{max_p} bar")
            else:
                min_p, max_p, default_p = 1, 10, 5
            
            print("\nEnter system pressure:")
            print("1. Low pressure")
            print("2. Medium pressure")
            print("3. High pressure")
            print("4. Custom pressure")
            
            pressure_choice = self.get_choice("Select option", 4)
            
            if pressure_choice == 1:
                pressure_bar = min_p
            elif pressure_choice == 2:
                pressure_bar = default_p
            elif pressure_choice == 3:
                pressure_bar = max_p
            else:
                pressure_bar = self.get_numeric_input(
                    f"Enter pressure (bar)", min_p, max_p, default_p
                )
            
            pressure_pa = pressure_bar * 1e5  # Convert to Pa
            
            # Show what saturation temperature this gives
            try:
                refprop = RefpropInterface()
                refprop.set_fluid(selected_refrigerant)
                sat_props = refprop.get_saturation_properties(P=pressure_pa)
                
                print(f"\n✅ System conditions:")
                print(f"   Pressure: {pressure_bar:.1f} bar")
                print(f"   Saturation temperature: {sat_props.T_sat:.1f}K ({sat_props.T_sat-273.15:.1f}°C)")
                print(f"   Latent heat: {sat_props.h_fg/1e6:.2f} MJ/kg")
                print(f"   Liquid density: {sat_props.rho_l:.1f} kg/m³")
                print(f"   Vapor density: {sat_props.rho_v:.2f} kg/m³")
            except Exception as e:
                print(f"   Unable to preview properties: {e}")
            
            return {
                'fluid': selected_refrigerant,
                'pressure': pressure_pa,
                'use_refprop': True
            }
    
    def get_vapor_chamber_params(self) -> Dict[str, Any]:
        """Get parameters for vapor chamber cooling"""
        self.clear_screen()
        self.print_header("VAPOR CHAMBER PARAMETERS")
        
        print("\nVapor chambers provide excellent heat spreading")
        print("They use two-phase heat transfer internally")
        
        # Vapor chamber configuration
        print("\n1. Vapor Chamber Configuration")
        print("\nAvailable configurations:")
        for i, vc_key in enumerate(self.vc_configs, 1):
            vc_data = self.data._dimensions_data['vapor_chamber_configs'][vc_key]
            thickness = vc_data['geometry']['chamber_thickness'] * 1000
            print(f"{i}. {vc_data['name']} - {thickness:.1f}mm thick")
        
        vc_choice = self.get_choice("Select vapor chamber", len(self.vc_configs))
        selected_vc = self.vc_configs[vc_choice - 1]
        
        # Show VC details
        vc_data = self.data._dimensions_data['vapor_chamber_configs'][selected_vc]
        print(f"\n✅ Selected: {vc_data['name']}")
        print(f"   Chamber thickness: {vc_data['geometry']['chamber_thickness']*1000:.1f} mm")
        print(f"   Evaporator area: {vc_data['geometry']['evaporator_area']*10000:.1f} cm²")
        print(f"   Condenser area: {vc_data['geometry']['condenser_area']*10000:.1f} cm²")
        
        # Condenser temperature
        print("\n2. Condenser Temperature")
        print("Temperature at the condenser side of vapor chamber")
        print("This depends on your secondary cooling method")
        
        T_condenser = self.get_temperature_input("Enter condenser temperature", default_celsius=25)
        
        params = {
            'vc_config': selected_vc,
            'T_condenser': T_condenser
        }
        
        print(f"\n✅ Vapor chamber parameters:")
        print(f"   Configuration: {selected_vc}")
        print(f"   T_condenser = {T_condenser:.1f} K ({T_condenser-273.15:.1f}°C)")
        
        return params
    
    def get_hybrid_cooling_params(self) -> Dict[str, Any]:
        """Get parameters for hybrid cooling"""
        self.clear_screen()
        self.print_header("HYBRID COOLING PARAMETERS")
        
        print("\nHybrid cooling combines multiple cooling methods")
        print("Example: Vapor chamber + Liquid cooling")
        
        print("\nSelect primary cooling method:")
        print("1. Vapor Chamber + Liquid")
        print("2. Liquid + Air (redundant)")
        print("3. Custom combination")
        
        hybrid_choice = self.get_choice("Select combination", 3)
        
        if hybrid_choice == 1:
            # Get vapor chamber params
            print("\n--- Primary: Vapor Chamber ---")
            vc_params = self.get_vapor_chamber_params()
            
            # Get liquid cooling params for secondary
            print("\n--- Secondary: Liquid Cooling ---")
            liquid_params = self.get_liquid_cooling_params()
            
            # Combine parameters
            params = {
                'primary_type': 'vapor_chamber',
                'secondary_type': 'liquid',
                **vc_params,
                **liquid_params
            }
        else:
            # Simplified for other options
            print("\nCustom hybrid configurations require manual setup")
            params = self.get_liquid_cooling_params()  # Default to liquid
        
        return params
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive configuration validation"""
        errors = []
        warnings = []
        
        # Check power
        if not 10 <= config['power_watts'] <= 1000:
            errors.append(f"Power {config['power_watts']}W outside reasonable range (10-1000W)")
        
        # Check temperature
        if not 250 <= config['T_ambient'] <= 350:
            errors.append(f"Ambient temperature {config['T_ambient']}K outside reasonable range")
        
        # Check layer stack if customized
        if self.layer_editor and self.layer_editor.modified:
            # Check for extreme thickness
            total_thickness = sum(layer.thickness for layer in self.layer_editor.layers)
            if total_thickness > 0.01:  # 10mm
                warnings.append(f"Total stack thickness {total_thickness*1000:.1f}mm is very large")
            
            # Check for area mismatches
            for i in range(len(self.layer_editor.layers) - 1):
                area_ratio = self.layer_editor.layers[i+1].area / self.layer_editor.layers[i].area
                if area_ratio > 4:
                    warnings.append(f"Large area increase ({area_ratio:.1f}x) between layers {i+1} and {i+2}")
                elif area_ratio < 0.25:
                    warnings.append(f"Large area decrease ({1/area_ratio:.1f}x) between layers {i+1} and {i+2}")
        
        # Cooling-specific validation
        if config['cooling_type'] == CoolingType.EVAPORATIVE:
            if config['cooling_params']['fluid'] not in self.fluids:
                errors.append(f"Invalid fluid: {config['cooling_params']['fluid']}")
        
        if config['cooling_type'] == CoolingType.VAPOR_CHAMBER:
            if config['cooling_params']['vc_config'] not in self.vc_configs:
                errors.append(f"Invalid vapor chamber config: {config['cooling_params']['vc_config']}")
        
        is_valid = len(errors) == 0
        messages = errors + warnings
        
        return is_valid, messages
    
    def save_configuration(self, config_name: str, config: Dict[str, Any]) -> bool:
        """Save custom configuration to JSON"""
        try:
            # Build configuration data
            config_data = {
                "name": config_name,
                "layers": []
            }
            
            # Use custom layers if available
            layers = self.layer_editor.layers if self.layer_editor else self.data.get_chip_configuration(config['config_key'])
            
            for layer in layers:
                config_data["layers"].append({
                    "name": layer.name,
                    "thickness": layer.thickness,
                    "area": layer.area,
                    "material": layer.material_key,
                    "notes": layer.notes
                })
            
            # Save configuration
            self.data.save_custom_configuration(config_name, config_data)
            
            # Save custom materials if any
            for mat_key, mat_data in self.material_editor.custom_materials.items():
                self.data.save_custom_material(mat_key, mat_data)
            
            print(f"✅ Configuration '{config_name}' saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False
    
    def show_configuration_summary(self, config: Dict[str, Any]):
        """Show detailed configuration summary"""
        self.print_header("CONFIGURATION SUMMARY")
        
        # Basic parameters
        print(f"\nBase Configuration: {config['config_key']}")
        if self.layer_editor and self.layer_editor.modified:
            print("  ✏️  Layer stack has been customized")
        
        print(f"\nPower Dissipation: {config['power_watts']} W")
        print(f"Ambient Temperature: {config['T_ambient']:.1f} K ({config['T_ambient']-273.15:.1f}°C)")
        print(f"Cooling Type: {config['cooling_type'].value}")
        
        # Layer stack summary
        if self.layer_editor:
            print(self.layer_editor.get_stack_summary())
        
        # Cooling parameters
        print("\nCooling Parameters:")
        for param, value in config['cooling_params'].items():
            if isinstance(value, float):
                if 'T_' in param:  # Temperature parameter
                    print(f"  {param}: {value:.1f} K ({value-273.15:.1f}°C)")
                elif param.startswith('A_') or '_area' in param:  # Area parameter
                    print(f"  {param}: {value:.4f} m² ({value*10000:.1f} cm²)")
                else:
                    print(f"  {param}: {value}")
            else:
                print(f"  {param}: {value}")
        
        # Custom materials
        if self.material_editor.custom_materials:
            print(f"\nCustom Materials: {len(self.material_editor.custom_materials)}")
            for mat_key in self.material_editor.custom_materials:
                print(f"  - {mat_key}")
    
    def final_workflow_menu(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Present final workflow options"""
        self.clear_screen()
        self.print_header("CONFIGURATION COMPLETE")
        
        # Validate configuration
        is_valid, messages = self.validate_configuration(config)
        
        if not is_valid:
            print("\n❌ CONFIGURATION ERRORS:")
            for msg in messages:
                print(f"  - {msg}")
            print("\nPlease fix errors before proceeding")
            input("\nPress Enter to go back...")
            return None
        
        if messages:  # Warnings only
            print("\n⚠️  WARNINGS:")
            for msg in messages:
                print(f"  - {msg}")
        
        # Show summary
        self.show_configuration_summary(config)
        
        print("\n" + "="*60)
        print("What would you like to do next?")
        print("1. Run thermal analysis now")
        print("2. Save configuration to JSON and run analysis")
        print("3. Save configuration to JSON and exit")
        print("4. Export configuration report")
        print("5. Go back and modify")
        print("6. Exit without saving")
        
        choice = self.get_choice("Select action", 6)
        
        if choice == 1:
            # Run analysis directly
            return config
            
        elif choice == 2:
            # Save and run
            config_name = input("\nEnter configuration name: ").strip()
            if not config_name:
                config_name = f"custom_config_{self.current_config}"
            
            if self.save_configuration(config_name, config):
                config['config_key'] = config_name
                return config
            else:
                return self.final_workflow_menu(config)
                
        elif choice == 3:
            # Save and exit
            config_name = input("\nEnter configuration name: ").strip()
            if not config_name:
                config_name = f"custom_config_{self.current_config}"
            
            self.save_configuration(config_name, config)
            print("\nConfiguration saved. Exiting...")
            return None
            
        elif choice == 4:
            # Export report
            self.export_configuration_report(config)
            return self.final_workflow_menu(config)
            
        elif choice == 5:
            # Go back
            return self.run_interactive_guide()
            
        else:
            # Exit without saving
            if self.config_modified:
                confirm = input("\nDiscard all changes? (y/n): ").strip().lower()
                if confirm != 'y':
                    return self.final_workflow_menu(config)
            
            print("Exiting without saving...")
            return None
    
    def export_configuration_report(self, config: Dict[str, Any]):
        """Export detailed configuration report"""
        report_name = f"thermal_config_report_{config['config_key']}.txt"
        
        try:
            with open(report_name, 'w') as f:
                f.write("GPU THERMAL ANALYSIS CONFIGURATION REPORT\n")
                f.write("="*60 + "\n\n")
                
                # Basic info
                f.write(f"Configuration: {config['config_key']}\n")
                f.write(f"Power: {config['power_watts']} W\n")
                f.write(f"Ambient Temperature: {config['T_ambient']:.1f} K ({config['T_ambient']-273.15:.1f}°C)\n")
                f.write(f"Cooling Type: {config['cooling_type'].value}\n\n")
                
                # Layer stack details
                f.write("LAYER STACK DETAILS\n")
                f.write("-"*60 + "\n")
                
                layers = self.layer_editor.layers if self.layer_editor else self.data.get_chip_configuration(config['config_key'])
                
                for i, layer in enumerate(layers, 1):
                    f.write(f"\nLayer {i}: {layer.name}\n")
                    f.write(f"  Thickness: {layer.thickness*1000:.3f} mm\n")
                    f.write(f"  Area: {layer.area*1e6:.1f} mm²\n")
                    f.write(f"  Material: {layer.material_key}\n")
                    
                    try:
                        mat = self.data.get_material(layer.material_key)
                        f.write(f"  Thermal conductivity: {mat.k} W/m·K\n")
                        f.write(f"  Density: {mat.rho} kg/m³\n")
                        f.write(f"  Specific heat: {mat.cp} J/kg·K\n")
                    except:
                        pass
                
                # Cooling parameters
                f.write("\n\nCOOLING PARAMETERS\n")
                f.write("-"*60 + "\n")
                for param, value in config['cooling_params'].items():
                    f.write(f"{param}: {value}\n")
                
                # Modifications
                if self.layer_editor and self.layer_editor.modified:
                    f.write("\n\nMODIFICATIONS FROM BASE CONFIGURATION\n")
                    f.write("-"*60 + "\n")
                    # Add modification details
                
                f.write("\n\nReport generated successfully.\n")
            
            print(f"✅ Report exported to: {report_name}")
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")


def main():
    """Enhanced command line interface"""
    guide = AdvancedThermalInputGuide()
    
    print("ADVANCED GPU THERMAL ANALYSIS CONFIGURATION TOOL")
    print("="*60)
    print("Granular customization of thermal analysis parameters")
    print("Modify layer stacks, materials, and cooling configurations")
    
    result = guide.run_interactive_guide()
    
    if result:
        # Run thermal analysis if requested
        if 'config_key' in result and result['config_key']:
            print("\n" + "="*60)
            print("RUNNING THERMAL ANALYSIS")
            print("="*60)
            
            calc = JunctionToSurfaceCalculator()
            
            # Load configuration with custom layers if available
            if result.get('custom_layers'):
                # Load base configuration first
                calc.load_configuration(result['config_key'].replace('custom_config_', ''))
                # Then replace layers
                calc.layers = []
                for layer_info in result['custom_layers']:
                    material = calc.data.get_material(layer_info.material_key)
                    calc.layers.append(ThermalLayer(layer_info, material))
            else:
                calc.load_configuration(result['config_key'])
            
            # Check if we need power model
            use_power_model = input("\nUse temperature-dependent power model? (y/n) [n]: ").strip().lower()
            
            if use_power_model == 'y':
                # Get power model parameters
                print("\nPower Model Parameters:")
                V_dd = guide.get_numeric_input("Supply voltage (V)", 0.5, 2.0, 1.1)
                f_clock = guide.get_numeric_input("Clock frequency (GHz)", 0.5, 4.0, 2.0) * 1e9
                
                power_model = PowerModel(
                    V_dd=V_dd,
                    f_clock=f_clock,
                    C_eff=1e-9,
                    alpha=0.7,
                    cores=3584
                )
                
                results = calc.solve_nonlinear(
                    power_model,
                    result['T_ambient'],
                    result['cooling_type'],
                    result['cooling_params']
                )
            else:
                # Simple steady state analysis
                results = calc.calculate_steady_state(
                    result['power_watts'],
                    result['T_ambient'],
                    result['cooling_type'],
                    result['cooling_params']
                )
            
            # Show results summary
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"Junction Temperature: {results['temperatures']['T_junction']:.1f} K "
                  f"({results['temperatures']['T_junction']-273.15:.1f}°C)")
            print(f"Total Thermal Resistance: {results['resistances']['R_total']:.3f} K/W")
            
            # Performance assessment
            T_j = results['temperatures']['T_junction'] - 273.15
            if T_j < 85:
                status = "✅ EXCELLENT - Well within safe operating range"
            elif T_j < 95:
                status = "✅ GOOD - Normal operating temperature"
            elif T_j < 105:
                status = "⚠️  MARGINAL - Consider better cooling"
            else:
                status = "❌ POOR - Exceeds safe operating temperature"
            
            print(f"\nThermal Performance: {status}")
            
            # Generate plot
            plot_choice = input("\nGenerate temperature profile plot? (y/n) [y]: ").strip().lower()
            if plot_choice != 'n':
                calc.plot_temperature_profile(results)
    
    print("\nThank you for using the Advanced GPU Thermal Analysis Tool!")


if __name__ == "__main__":
    main()
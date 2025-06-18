from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as req_file:
        return [line.strip() for line in req_file 
                if line.strip() and not line.startswith('#')]

setup(
    name="data-card-calculator",
    version="0.1.0",
    description="",
    author="Johnson Controls Team",
    packages=find_packages(include=['main', 'main.*']),
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },
    # Include additional data files if needed
    include_package_data=True,
    package_data={
        'main': ['**/*.json', '**/*.yaml', '**/*.yml'],
        '': ['*.txt', '*.md'],
    }
)
"""import importlib
import subprocess

def check_module(module_name):
    try:
        # Try to import the module
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        # If the module is not found, print an error message
        print(f"The module '{module_name}' is not installed.")
    else:
        # If the module is found, use pip to get information about it
        result = subprocess.run(['pip', 'show', module_name], stdout=subprocess.PIPE)

        # Decode the output of the pip command
        output = result.stdout.decode('utf-8')

        # Split the output into lines and print the information
        lines = output.split('\n')
        print("---------------------------------------------")
        print(f"Information about module {module_name}:")
        for line in lines:
            print(f"    {line}")
        print("---------------------------------------------")

if __name__ == '__main__':
    
    module_name = input('Enter the name of a module: ')
    check_module(module_name)"""

# The below code provides a loop that helps scrutinze the modules one after the other

import importlib
import subprocess

while True:
    module_name = input("Enter the name of a module (enter 'exit' to end): ")
    if module_name == "exit":
        break

    try:
        # Try to import the module
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        # If the module is not found, print an error message
        print(f"The module '{module_name}' is not installed.")
    else:
        # If the module is found, use pip to get information about it
        result = subprocess.run(["pip", "show", module_name], stdout=subprocess.PIPE)

        # Decode the output of the pip command
        output = result.stdout.decode("utf-8")

        # Split the output into lines and print the information
        print("---------------------------------------------")
        print(f"Information about module {module_name}:")
        for line in output.split("\n"):
            print(f"    {line}")
        print("---------------------------------------------")
# This is a comment in Python - it helps explain what the code does
print("Welcome to PetGPT Learning!")

# Variables - storing different types of data
pet_name = "Buddy"               # String (text)
pet_age = 5                      # Integer (whole number)
pet_weight = 12.5               # Float (decimal number)
is_vaccinated = True            # Boolean (True/False)

# Print these variables
print("Pet Name:", pet_name)
print("Pet Age:", pet_age)
print("Pet Weight:", pet_weight)
print("Vaccinated:", is_vaccinated)

# Basic list of pets
pets = ["dog", "cat", "hamster", "parrot"]
print("\nTypes of pets:", pets)
print("First pet in list:", pets[0])  # Lists start counting from 0

# Simple if-else statement
if pet_age < 1:
    print(f"{pet_name} is a puppy!")
else:
    print(f"{pet_name} is an adult dog!")
# Function to calculate BMI
def calculate_bmi(weight, height):
    """Calculates BMI from weight in kg and height in cm."""
    return weight / (height / 100) ** 2

# Function to calculate BMR using the Mifflin-St Jeor Equation
def calculate_bmr(age, weight, height, gender):
    """Calculates BMR using the Mifflin-St Jeor Equation."""
    if gender.lower() == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']


squares = [n**2 for n in range(10)]
short_planets = [planet for planet in ['a','b','c','d','e'] if len(planet) < 6]


result = [
    planet.upper() + '!' 
    for planet in planets 
    if len(planet) < 6
]
print(result)

print(sum([len(planet) for planet in planets]))


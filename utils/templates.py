tags = """
- Alarm System
- Broadband
- Built in Robes
- Dishwasher
- Ensuite
- Fireplace
- Floor Boards
- Gym
- Heating Other
- Hot Water Gas
- Hot Water Electric
- Hot Water Solar
- Intercom
- Pay TV
- Rumpus Room
- Spa
- Study
- Vacuum System
- Workshop
- Balcony
- Courtyard
- Deck
- Fully Fenced
- Outdoor Entertainment
- Pool (Above Ground)
- Pool (Inground)
- Remote Garage
- Secure Parking
- Shed
- Tennis Court
- Air Conditioning
- Ducted Cooling
- Ducted Heating
- Evaporative Cooling
- Heating Gas
- Heating Electric
- Heating Hydronic
- Reverse Cycle Air Con
- Split System (Heating)
- Split System (Air Con)
- Solar Panels
- Water Tank
"""

template_1 = """
Description: "{description}"

Tags List:  ```{tags}```

For the above given description of a real estate property and the desired tags list, \
you need to extract appropriate tags that matches to the tags list given above. \
You need to find tags in such a way that those feature belong to the property \
and the extracted tags must be from the tags list only. Your output should in \
the form of array.

"""

template_1_1 = """
Description: "{description}"

Tags List:  ```{tags}```

For the above given description of a real estate property and the desired tags list, \
you need to extract tags that matches to the tags list given above. \
You need to find tags in such a way that those feature belong to the property and not \
in any near by facilities or services. Also the extracted tags must be from the tags list only. \
Your output should in the form of array.

"""

template_1_2 = """
Description: "{description}"

Tags List:  ```{tags}```

For the above given description of a real estate property and the desired tags list, \
you need to extract tags that matches to the tags list given above. Some examples are shown below \

Examples:
"Walk-in robe" -> "Built in Robes"
"Heating" -> "Heating Other"
"Pool" -> "Pool (Inground)"
"Ducted reverse cycle air conditioning" -> "Reverse Cycle Air Con", "Ducted Cooling"
"Underfloor heating and heated towel rails to ensuite" -> "Heating Gas", "Ensuite"
"garage/workshop" -> "Workshop"

This are just a few examples to make you understand.
You need to find tags in such a way that those feature belong to the property and not \
in any near by facilities or services. Also the extracted tags must be from the tags list only. \
Your output should in the form of array.

"""


template1 = """
Description: "{description}"

Tags List:  ```{tags}```

You are given two things i.e. the description of a real estate property and \
the tags that I desire to extract from a real estate property. Your task is \
to go through the description and extract tags that matches the corresponding tags \
in the tags list. Note that the tags you extract should be belonging to the property itself.
Your output should in the form of array. Please do not extract any other tags out of \
the tags given to you in the tags list strictly.
"""



template2 = """
Description: "{description}"

For the given description for a real estate property, extract the feature tags \
only from the pool of tags given to you below. Your output should in the form of array.

Tags List:  ```{tags}```
"""
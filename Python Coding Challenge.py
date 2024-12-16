

"""
These are two coding challenges using simple coding skills, like dictionary of dictionary, or several packages solving equations and inequitions.
"""
def gallery(visits, option=2):
    """
    Produce summary statistics of gallery visits.

    Parameters:
        visits: list of visits (see also HTML instructions):
            Each visit is a tuple (room number (str), visitor number (str), time (str)) (all elements are integers in string format)
            Each visitor starts outside any room, and they leave all rooms in the end.
            The visits are not necessarily in order.
        option (int, optional): determines what to return, see below
            
    Returns:
        a list containing tuples for each room (sorted in increasing number by room number (1, 2, 3, ...)):
        - if option = 0, (room number, number of unique visitors)
        - if option = 1, (room number, number of unique visitors, average visit time)
        - if option = 2, (room number, number of unique visitors, average visit time, highest total time spent in the room by a single visitor)
        - the average visit time is rounded to integer value.

    Example use:
    >>> visits = [('0', '0', '20'), ('0', '0', '25'), ('1', '1', '74'), ('1', '1', '2')]
    >>> gallery(visits)
    [('0', 1, 5, 5), ('1', 1, 72, 72)]
    >>> gallery(visits, 0)
    [('0', 1), ('1', 1)]
    >>> gallery(visits, 1)
    [('0', 1, 5), ('1', 1, 72)]
    >>> gallery(visits, 1)[0]
    ('0', 1, 5)
    >>> visits = [('15', '3', '61'), ('15', '3', '45'), ('6', '0', '91'), ('10', '4', '76'), ('6', '0', '86'), ('6', '4', '2'), ('10', '1', '47'), ('6', '3', '17'), ('6', '4', '41'), ('15', '3', '36'), ('6', '2', '97'), ('15', '4', '58'), ('6', '0', '16'), ('10', '2', '21'), ('10', '4', '75'), ('6', '0', '76'), ('15', '4', '50'), ('10', '1', '64'), ('6', '3', '3'), ('15', '3', '35'), ('6', '2', '96'), ('10', '2', '35'), ('10', '2', '77'), ('10', '2', '48')]
    >>> gallery(visits)
    [('6', 4, 24, 65), ('10', 3, 15, 43), ('15', 2, 8, 17)]
    """
    # Your code here. Don't change anything above.
    visits_int = [(int(room), int(visitor), int(time)) for room, visitor, time in visits]
    sorted_visits=sorted(visits_int)
    diff_visits = []
    for i in range(0, len(sorted_visits), 2):
        if i + 1 < len(sorted_visits):  
            first = sorted_visits[i]
            second = sorted_visits[i + 1]
            diff = int(second[2]) - int(first[2])
            diff_visits.append((second[0], second[1], diff))
    new_visit = [(str(room), str(visitor), int(time)) for room, visitor, time in diff_visits]
    room_data={}
    for visit in new_visit:
        room, visitor, time = visit  
        time = int(time)
        
        if room not in room_data:
            room_data[room] = {'visitors': {}, 'total_time': 0, 'max_time': 0 }
            
        if visitor not in room_data[room]['visitors']:
            room_data[room]['visitors'][visitor] = {'total_time':0,'count_visit':0}

        room_data[room]['visitors'][visitor]['total_time'] += time
        room_data[room]['visitors'][visitor]['count_visit']+= 1
        room_data[room]['total_time'] += time
        room_data[room]['max_time'] = max(room_data[room]['max_time'], room_data[room]['visitors'][visitor]['total_time'])
    results = []
    for room in sorted(room_data.keys(), key=int):
        unique_visitors = len(room_data[room]['visitors'])
        total_visits = sum(room_data[room]['visitors'][visitor]['count_visit'] for visitor in room_data[room]['visitors'])
        average_time = round(room_data[room]['total_time'] / total_visits)
        max_time = room_data[room]['max_time']
        
        if option == 0:
            results.append((room, unique_visitors))
        elif option == 1:
            results.append((room, unique_visitors, average_time))
        elif option == 2:
            results.append((room, unique_visitors, average_time, max_time))
    
    return results
        
    
    
    pass


def reverse_engineer(seq):
    """
    Reverse engineer an input sequence
    
    Parameters:
        seq - list of strings
    
    Returns:
        list of values corresponding to each letter present in the sequences (smallest possible values)
        (in alphabetical order)
    
    Example use
    >>> reverse_engineer(["a", "ab", "c", "a", "ab", "ac"])
    [2, 4, 5]
    >>> reverse_engineer(["b", "bc", "ab", "bc", "b", "abc", "b"])
    [3, 1, 2]
    >>> reverse_engineer(["a", "b", "d", "c", "a", "ab"])
    [6, 9, 11, 10]
    >>> reverse_engineer(['c', 'ce', 'd', 'c', 'ce', 'd', 'c', 'a', 'ce', 'cd', 'b', 'ce', 'c', 'd', 'ce', 'c', 'a', 'd', 'ce', 'c', 'cde', 'c', 'b', 'ce', 'd', 'ac', 'ce', 'd', 'c', 'ce', 'cd', 'ce', 'a', 'bc', 'd', 'ce', 'c', 'd', 'ce', 'c', 'cde', 'a', 'c', 'ce', 'df', 'b', 'c', 'ce', 'd', 'c', 'ace', 'cd', 'ce', 'c', 'd', 'ce', 'b', 'c', 'ad', 'ce', 'c'])
    [17, 23, 3, 7, 6, 91]
    """
    # Your code here. Don't change anything above.
    import re
    from collections import defaultdict
    def extract_unique_letters(sequence):
        unique_letters = []
        seen = set()
        
        for item in sequence:
            for letter in item:
                if letter not in seen:
                    seen.add(letter)
                    unique_letters.append(letter)
        
        return unique_letters
    def generate_dataset(sequence):
        letter_position = []
        letter_count = defaultdict(int)
        
        for idx, item in enumerate(sequence):
            pos = idx + 1
            for letter in item:
                letter_count[letter] += 1
                letter_position.append((letter, pos, letter_count[letter]))
        
        return letter_position
    def generate_equality_formulas(dataset):
        position_map = defaultdict(list)
        
        # Group letters by position
        for letter, pos, occurrence in dataset:
            position_map[pos].append((letter, occurrence))
        
        equality_formulas = []
        
        
        for pos, letters in position_map.items():
            if len(letters) > 1:
                formula = ""
                left_side = []
                for letter, occ in letters:
                    left_side.append(f"{occ}{letter}")
                formula += "=".join(left_side)
                equality_formulas.append(formula)
        
        return equality_formulas
    def remove_shared_positions_and_generate_inequality(dataset):
        position_map = defaultdict(list)
        
       
        for letter, pos, occurrence in dataset:
            position_map[pos].append((letter, occurrence))
        
      
        filtered_dataset = [entry for entry in dataset if len(position_map[entry[1]]) == 1]
        
        
        remaining_letters = sorted(filtered_dataset, key=lambda x: (x[1], x[2]))  
        inequality = []
        for letter, _, occurrence in remaining_letters:
            if occurrence == 1:
                inequality.append(letter)
            else:
                inequality.append(f"{occurrence}{letter}")
        
        return "<".join(inequality)
    def parse_inequality_formula(inequality_formulas):
        inequality_relations = []
        for formula in inequality_formulas:
            variables = re.split(r'\s*<\s*', formula)
            inequality_relations.append(variables)
        return inequality_relations
    def parse_equality_formulas(equality_formulas):
        relations = []
        for formula in equality_formulas:
            parts = formula.split('=')
            
            for i in range(len(parts) - 1):
                left_side, right_side = parts[i], parts[i + 1]
                left_coeff, left_var = re.match(r'(\d*)([a-zA-Z]+)', left_side).groups()
                right_coeff, right_var = re.match(r'(\d*)([a-zA-Z]+)', right_side).groups()
                left_coeff = int(left_coeff) if left_coeff else 1
                right_coeff = int(right_coeff) if right_coeff else 1
                relations.append((left_var, left_coeff, right_var, right_coeff))
        return relations
    def apply_equality_relations(assignment, equality_relations):
        updated_assignment = assignment.copy()
        for left_var, left_coeff, right_var, right_coeff in equality_relations:
            if left_var in updated_assignment and right_var not in updated_assignment:
                value = (left_coeff * updated_assignment[left_var]) / right_coeff
                if not value.is_integer():
                    return None
                updated_assignment[right_var] = int(value)
            elif right_var in updated_assignment and left_var not in updated_assignment:
                value = (right_coeff * updated_assignment[right_var]) / left_coeff
                if not value.is_integer():
                    return None
                updated_assignment[left_var] = int(value)
            elif left_var in updated_assignment and right_var in updated_assignment:
                if (left_coeff * updated_assignment[left_var]) != (right_coeff * updated_assignment[right_var]):
                    return None
        return updated_assignment
    def eval_term(term, assignment):
        match = re.match(r'(\d*)([a-zA-Z]+)', term)
        if match:
            coeff, var = match.groups()
            coeff = int(coeff) if coeff else 1
            return coeff * assignment[var]
        return assignment[term]
    def satisfies_inequality(assignment, inequality):
        for i in range(len(inequality) - 1):
            left_term = inequality[i]
            right_term = inequality[i + 1]
            
            left_value = eval_term(left_term, assignment)
            right_value = eval_term(right_term, assignment)

            if left_value >= right_value:
                return False
        return True
    unique_letters = extract_unique_letters(seq)
    inequality_formula_1 = "<".join(unique_letters)

 
    dataset = generate_dataset(seq)

    
    equality_formulas = generate_equality_formulas(dataset)

 
    inequality_formula_2 = remove_shared_positions_and_generate_inequality(dataset)

   
    inequality_formulas = [inequality_formula_1, inequality_formula_2]
    equality_relations = parse_equality_formulas(equality_formulas)
    inequality_relations = parse_inequality_formula(inequality_formulas)

    smallest_var = inequality_relations[0][0]
    
    
    for initial_value in range(1, 101):  
        assignment = {smallest_var: initial_value}
        
       
        assignment = apply_equality_relations(assignment, equality_relations)
        if assignment is None:
            continue 
        
        
        for i in range(1, len(inequality_relations[0])):
            var = inequality_relations[0][i]
            if var not in assignment:
                assignment[var] = assignment[inequality_relations[0][i - 1]] + 1
        
      
        if satisfies_inequality(assignment, inequality_relations[1]):
            
            sorted_assignment = sorted(assignment.items())  
            return [value for _, value in sorted_assignment]

    return None  




    

    
    

    

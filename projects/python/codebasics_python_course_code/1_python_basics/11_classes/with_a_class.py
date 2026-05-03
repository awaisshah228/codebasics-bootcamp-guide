import datetime

class CricketPlayer:
    team_size = 11
    def __init__(self, fname, lname, birth_year, team):
        self.first_name = fname
        self.last_name = lname
        self.birth_year = birth_year
        self.team = team
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def get_average_score(self):
        return sum(self.scores)/len(self.scores)

    def get_age(self):
        now = datetime.datetime.now()
        return now.year - self.birth_year

    def __str__(self):
        return f"{self.first_name} {self.last_name}, the cricket player from {self.team}"

    def __lt__(self, other):
        self_avg_score = self.get_average_score()
        other_avg_score = other.get_average_score()
        return self_avg_score < other_avg_score

    def __eq__(self, other):
        self_age = self.get_age()
        other_age = other.get_age()
        return self_age == other_age

virat = CricketPlayer('virat', 'kohli', 1988, 'India')
virat.add_score(37)
virat.add_score(100)
virat.add_score(23)

david = CricketPlayer('david','warner', 1988, 'australia')
david.add_score(37)
david.add_score(23)
david.add_score(85)

print("Virat avg score: ",virat.get_average_score())
print("David avg score: ", david.get_average_score())

print("Is virat less than david: ", virat < david)
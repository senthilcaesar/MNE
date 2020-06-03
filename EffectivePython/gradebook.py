from collections import defaultdict
'''
Python's convention to make an instance variable protected is to add a 
prefix _ (single underscore) to it. This effectively prevents it to 
be accessed, unless it is from within a sub-class.
'''
class SimpleGradebook:
    def __init__(self):
        self._grades = {}

    def add_student(self, name):
        self._grades[name] = defaultdict(list)

    def report_grade(self, name, subject, score, weight):
        by_subject = self._grades[name]
        grade_list = by_subject[subject]
        grade_list.append((score, weight))

    def average_grade(self, name):
        by_subject = self._grades[name]
        score_sum, score_count = 0, 0
        for subject, scores in by_subject.items():
            subject_avg, total_weight = 0, 0
            for score, weight in scores:
                subject_avg += score * weight
                total_weight += weight
            score_sum += subject_avg / total_weight
            score_count += 1
        return score_sum / score_count

    def show_book(self, name):
        print(self._grades[name])

book = SimpleGradebook()
book.add_student('caesar')
book.report_grade('caesar', 'Math', 75, 0.05)
book.report_grade('caesar', 'Math', 65, 0.15)
book.report_grade('caesar', 'Math', 70, 0.80)
book.report_grade('caesar', 'Physics', 100, 0.40)
book.report_grade('caesar', 'Physics', 85, 0.60)
print(book.average_grade('caesar'))
book.show_book('caesar')
from datafc.syntactic.token import TokenData


class Validator:
    def __init__(self, threshold):
        self.threshold = threshold

    def validate(self, transformed_values, target_values, scores):
        print(self.validate_syntactic(transformed_values, target_values))
        print(self.validate_semantic(scores))
        return self.validate_syntactic(transformed_values, target_values) or self.validate_semantic(scores)

    def validate_syntactic(self, transformed_values, target_values):
        token_lists1 = []
        for str_value in transformed_values:
            token_lists1.append(" ".join([x.name for x in TokenData.get_basic_tokens(str_value)]))

        token_lists2 = []
        for str_value in target_values:
            token_lists2.append(" ".join([x.name for x in TokenData.get_basic_tokens(str_value)]))

        if (set(token_lists1)).issubset((set(token_lists2))):
            return True
        return False

    def validate_semantic(self, scores):
        for score in scores:
            if score >= self.threshold:
                return True
        return False

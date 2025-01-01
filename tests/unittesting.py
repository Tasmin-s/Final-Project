import unittest
from function import multiply_two_numbers

class TestFunction(unittest.TestCase):
    
    def test_multiplication(self):
        result = multiply_two_numbers(3,5)
        self.assertEqual(result, 15)

    def test_multiplication_negative_numbers(self):
        result = multiply_two_numbers(-3,5)
        self.assertEqual(result, -15)

    def test_multiplication_two_negative_numbers(self):
        result = multiply_two_numbers(-3,-5)
        self.assertEqual(result, 15)

    def test_multiplication_integer(self):
        with self.assertRaises(TypeError):
            multiply_two_numbers("1",1)
            multiply_two_numbers(1,"2")

if __name__ == '__main__':
    unittest.main()
    



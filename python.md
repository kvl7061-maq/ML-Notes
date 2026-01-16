# General and important rules python specific

* There is no length limit for identifiers in python
* **Interning** is an optimization where Python reuses certain objects
* Once we create an object, we can not perform any changes in that object if we are trying to perform any changes with those changes a new object will be created and we can verify it using id().
* _x : Protected variable
* __x : Private variable
* \_\_x\_\_ : It is valled magic variable
* Keywords :
  * True, False, None keywords in this first letter is capital.
  is, and, or, not, yield, in etc.. are keyword in python
  * **In python its try,except finally and not try, catch finally**
  * Gotcha: and/or return one operand (not necessarily True/False)
  * yield turns the function into a generator
  * lambda is an anonymous function (limited to 1 expression)
  * Gotcha: with ensures cleanup even on exceptions.
  * There is no switch statement concept in python instead in higher version of python there is similar concept called match case
  * do - while loop concept is not there in python
  * int,float,char etc are not reserved words in python like other languages
  * pass → placeholder statement
  * del → delete reference
  * is → identity comparison
  * in → membership test
  * **== checks if two objects have the same value, while "is" checks if they are the exact same object in memory.**
* ⚠️ Numeric Gotchas
  * dict/set require hashable keys/elements
  * float precision :    0.1 + 0.2 == 0.3   # False, Because floats are stored in binary approximation.
  * Immutable :
    * int, float, bool, str, tuple, frozenset, bytes
  * Mutable :
    * list, dict, set, bytearray
* ⚠️ String Gotchas  
  * In python there is no char data type
  * We can use single quote, double quote as well as triple quotes in python for defining stirngs. The tripe quotes (''' or """) is used to define multiline stirng literal
  * s.find("py")      # returns index or -1
  * s.index("py")     # returns index or ValueError if not found
  * Use raw strings for paths and raw string cannot end with a single backslash
    * path = r"C:\new\test"
  * splitlines(), endswith(), startswith() are Very useful in file processing:
    * text.splitlines()
    * filename.endswith(".csv")
    * s.startswith("http")
  * slicing doesn’t throw index errors
  * while concatenating string using ' + ' we should ensure that both variable should be of string type.
  * "*" operator for string act as string repetation operator.

* ⚠️ String Gotchas  :
  * Syntax : seq[start : stop : step]
  * If step is negative:
    * with step -1, slicing expects start > stop.
    * default start becomes len(seq)-1
    * default stop becomes -1 (conceptually “before index 0” that why it also return the value present at 0 index.)
* List :
  * A Python list is:
    * an ordered collection
    * duplicates are allowed
    * mutable (can be changed in place)
    * can hold mixed data types
    * Slicable
  * Functions :
    * append() : adds single element at end.
    * extend() : adds multiple elements at end.
    * insert() : at any locations
    * remove()
    * index()
    * count()
    * enumerate() : Iterate with index
    * zip()
* Tuple :
  * Same as list except that it is immutable in nature.
  * t=() # Type = tuple
  * t=(10) Type = int
  * t=(10,) Type = tuple
 *⚠️ Type Conversion Gotchas  :
  * int() cannot directly convert strings containing a decimal point. Becacuse internally the string value should be in base 10 format only.
  * while using float() function for converting string to float internally the string value should be either integral value or float and should be specified in base 10 format only.
  * bool() :
    * bool(10) : True # Non zero value always results to true.
    * bool(0) : False
    * bool(0.0) : False
    * bool('') : False
    * bool('True') : True # Because only empty string is considered as false.

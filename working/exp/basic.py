str = "Hello"
print(str[-4:-2:2])
print("~".join(["AAA", "BBB", "CCC"]))
print("~".join(["AAA", "BBB", "CCC"])*3)
print(range(10))

list1=["a", "aa", "b", "bb"]
dict1={"a": "aa", "b": "bb"}
set1={"a", "aa", "b", "bb"}
set2=set(["a", "aa", "b", "bb"])
tuple1=("a", "aa", "b", "bb")


print(type(list1))
print(type(dict1))
print(type(set1))
print(type(set2))
print(type(tuple1))



for i in range(30):
    if i > 20:
        print("ok")
        break
else:
    print("false")



for key in dict1:
    print(key)

for key, value in dict1.items():
    print(key, value)

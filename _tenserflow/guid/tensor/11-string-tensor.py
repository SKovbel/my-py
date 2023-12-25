import tensorflow as tf
import numpy as np

scalar_string_tensor = tf.constant("Gray wolf")
tensor_of_strings = tf.constant(["Gray wolf", "Quick brown fox", "Lazy dog"])
tensor_unicode = tf.constant("🥳👍")
text = tf.constant("1 10 100")
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)


print(scalar_string_tensor)
print(tensor_of_strings)
print(tensor_unicode)
print(tf.strings.split(scalar_string_tensor, sep=" "))
print(tf.strings.split(tensor_of_strings))
print(tf.strings.to_number(tf.strings.split(text, " ")))
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)


# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)


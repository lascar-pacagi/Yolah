# cmake -DIN=file -DOUT=file.cpp -DNAME=symbol -P embed_source.cmake
# Writes `extern const char NAME[] = R"...(<contents of IN>)...";`
file(READ "${IN}" CONTENTS)
file(WRITE "${OUT}" "// generated from ${IN} by misc/embed_source.cmake\nextern const char ${NAME}[] = R\"YOLAH_SRC(${CONTENTS})YOLAH_SRC\";\n")

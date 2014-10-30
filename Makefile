CC       = mpicc
CFLAGS     = -c --std=c99

CLINKER  = mpicc
LDFLAGS    =  -lm

.SUFFIXES: .c .o
.c.o:
	$(CC) $(CFLAGS) $<

SRC   = hpl.c

OBJ = $(SRC:.c=.o)

TARGET = hpl

$(TARGET): $(OBJ)
	$(CLINKER) $(OBJ) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(OBJ) $(TARGET) core

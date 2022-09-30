#BOOST = "/home/and/boost_1_63_0"
#LIB64 = "/usr/lib"
#STD64 = "/usr/lib/gcc/x86_64-linux-gnu/5.4.0"
#release variables
FLAGS = -O3 -DNDEBUG #-DBOOST_SYSTEM_STATIC_LINK=1  -DDATE_TIME_INLINE
LINKING = -L"/usr/lib" -lpthread -lrt -L"/usr/lib/gcc/x86_64-linux-gnu/5.4.0" -lstdc++ -L"/home/hulaa/boost_1_68_0/stage/lib" -lboost_thread -lboost_random -lboost_system
#-lboost_thread -lboost_random -DBOOST_THREAD_USE_LIB=1 / 
#debug variables  
#FLAGS = -O0 -g -DBOOST_SYSTEM_STATIC_LINK=1 -DBOOST_THREAD_USE_LIB=1 -DDATE_TIME_INLINE
#LINKING = -L"C:/mingw/lib" -lboost_thread-mgw47-mt-d-1_53 -lboost_random-mgw47-mt-d-1_53 -lboost_system-mgw47-mt-d-1_53

SOURCES =  global.cpp  main.cpp #RandomSingleton.cpp main.cpp
INCLUDES = -I"/home/hulaa/boost_1_68_0"
#MultiWorker.cpp

EXECUTABLE = Est#.exe
OBJ_DIR = objects

.PHONY: all clean
.SILENT:

OBJECTS = $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(basename $(notdir $(SOURCES)))))
DEPFILES = $(OBJECTS:.o=.d)

all: $(EXECUTABLE)

-include $(DEPFILES)

$(EXECUTABLE): $(OBJ_DIR) $(OBJECTS)
	@echo \>\> linking $(EXECUTABLE)
	g++ -std=c++11 $(OBJECTS) $(FLAGS) -o $(EXECUTABLE) $(LINKING)
#-lboost_filesystem-mgw47-mt-1_53

$(OBJ_DIR):
	mkdir $@

$(OBJECTS):
	g++ -std=c++11 $(filter $(basename $(@F)).%, $(SOURCES)) $(FLAGS) $(INCLUDES) -MMD -MF $(@:.o=.d) -MT $@ -c -o $@ #

clean:
	del $(OBJ_DIR)
	del $(EXECUTABLE)

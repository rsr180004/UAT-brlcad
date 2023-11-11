#include "NeuralRayTracer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>

#include <pthread.h>
pthread_mutex_t model_mutex = PTHREAD_MUTEX_INITIALIZER;

NeuralRayTracer* NeuralRayTracer_Create(const char* model_path) {

    printf("CREATING A NRT!\n");

    NeuralRayTracer* instance = malloc(sizeof(NeuralRayTracer));
    
    //printf("NRT has been created!\n");
    PyEval_InitThreads();
    Py_Initialize();
    //printf("initial2!\n");
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../src/rt')");

    char file_path[256];
    snprintf(file_path, sizeof(file_path), "model_weights/%s_weights.pth", model_path);
    FILE* file = fopen(file_path, "r");
    char file_path_2[256];
    snprintf(file_path_2, sizeof(file_path_2), "model_weights/%s_hits_weights.pth", model_path);
    FILE* file2 = fopen(file_path_2, "r");

    instance->pName = PyUnicode_DecodeFSDefault("neural_trainer");

    instance->pModule = PyImport_Import(instance->pName);
    Py_XDECREF(instance->pName);

    if(instance->pModule != NULL) {
        printf("in file\n");
        instance->pModelClass = PyObject_GetAttrString(instance->pModule, "HitClassifier");
        if (PyCallable_Check(instance->pModelClass)) {  
            printf("in class\n");
            // This should never be the case (model should already get trained)
            if(!file) {
                printf("Can't locate model #1 for this specific database and object.\n");
                exit(EXIT_FAILURE);
            } else {
                printf("theres a model alive\n");

                // Instantiate CustomResNet first
                PyObject* pArgs = PyTuple_Pack(1, PyLong_FromLong(5)); // Assuming input size is 5
                //printf("calling init\n");
                instance->pModelInstance = PyObject_CallObject(instance->pModelClass, pArgs);


                //printf("pModelInstance in creator: %p\n", (void*)instance->pModelInstance);
                //printf("called init\n");
                Py_XDECREF(pArgs);

                printf("The model is going to be trained on the loaded weights\n");

                // Print the weights before and after loading weights
                //printf("WEIGHTS BEFORE\n");

                PyObject *pLoadWeightsMethod = PyObject_GetAttrString(instance->pModelInstance, "load_model1_weights");
                if (PyCallable_Check(pLoadWeightsMethod)) {
                    //printf("callable\n");
                    PyObject *p_newArgs = PyTuple_Pack(1, PyUnicode_FromString(file_path));
                    PyObject_CallObject(pLoadWeightsMethod, p_newArgs);
                    Py_XDECREF(p_newArgs);
                } 
                Py_XDECREF(pLoadWeightsMethod);

                printf("The models are now trained on the loaded weights\n");
                fclose(file);
            }
        }

        /*
        instance->pModelClassReg = PyObject_GetAttrString(instance->pModule, "Distance_Az_El_Model");
        if (PyCallable_Check(instance->pModelClassReg)) {  
            printf("in class\n");
            // This should never be the case (model should already get trained)
            if(!file2) {
                printf("Can't locate model #2 for this specific database and object.\n");
                exit(EXIT_FAILURE);
            } else {
                printf("theres a model 2 alive\n");

                // Instantiate CustomResNet first
                PyObject* pArgs = PyTuple_Pack(1, PyLong_FromLong(5)); // Assuming input size is 5
                //printf("calling init\n");
                instance->pModel2Instance = PyObject_CallObject(instance->pModelClassReg, pArgs);


                //printf("pModelInstance in creator: %p\n", (void*)instance->pModelInstance);
                //printf("called init\n");
                Py_XDECREF(pArgs);

                printf("Model 2 is going to be trained on the loaded weights\n");

                // Print the weights before and after loading weights
                //printf("WEIGHTS BEFORE\n");

                PyObject *pLoadWeightsMethod = PyObject_GetAttrString(instance->pModel2Instance, "load_model2_weights");
                if (PyCallable_Check(pLoadWeightsMethod)) {
                    //printf("callable\n");
                    PyObject *p_newArgs = PyTuple_Pack(1, PyUnicode_FromString(file_path_2));
                    PyObject_CallObject(pLoadWeightsMethod, p_newArgs);
                    Py_XDECREF(p_newArgs);
                } 
                Py_XDECREF(pLoadWeightsMethod);

                printf("The models are now trained on the loaded weights\n");
                fclose(file2);
            }
        }
        */
        //Py_XDECREF(instance->pModelClass); --> This caused the issue of the function not being callable in the shoot ray
        
    }
    Py_XDECREF(instance->pModule);
    

    return instance;
     
}

void NeuralRayTracer_Destroy(NeuralRayTracer* instance) {
    Py_XDECREF(instance->pModelInstance);
    // Py_XDECREF(instance->pModel2Instance);
    Py_XDECREF(instance->pModelClass);
    Py_XDECREF(instance->pModelClassReg);
    Py_Finalize();
    pthread_mutex_destroy(&model_mutex);
    free(instance);
}

void NeuralRayTracer_ShootRay(NeuralRayTracer* instance, double origin[3], double az_el[2], double output[1]) {
    
    // THE SEG FAULT COMES FROM RUNNING THE FOLLOWING CODE ON MULTIPLE CPUS AT THE SAME TIME

    // printf("in shoot ray method\n");
    //printf("pModelInstance: %p\n", (void*)instance->pModelInstance);
    // Lock the mutex to ensure exclusive access to the model
    pthread_mutex_lock(&model_mutex);

    //printf("IN NEURAL RAY TRACER SHOOT RAY METHOD in cpu: %s\n", buffer);
    if (instance->pModelInstance != NULL) {

        //printf("before predict\n");
        // printf("pModelInstance: %p\n", (void*)instance->pModelInstance);
        int present = PyObject_HasAttrString(instance->pModelInstance, "predict_hit_or_miss");

        // printf("in here shootray 1\n");
        PyObject* pMethod = PyObject_GetAttrString(instance->pModelInstance, "predict_hit_or_miss");
        // printf("checking if callable!\n");
        if (PyCallable_Check(pMethod)) {
            // printf("its callable 2\n");

            PyObject* pArgs = PyTuple_Pack(2, 
                                           Py_BuildValue("[d,d,d]", origin[0], origin[1], origin[2]),
                                           Py_BuildValue("[d,d]", az_el[0], az_el[1]));

            // printf("about to call!\n");
            
            PyObject* pResult = PyObject_CallObject(pMethod, pArgs);

            // printf("called\n");

            
            if (PyList_Check(pResult) && PyList_Size(pResult) == 1) {
                //double values[6];
                //printf("down here\n")
                // printf("---1\n");;
                for (int i = 0; i < 1; i++) {
                    //printf("down here2\n");
                    // printf("---2\n");
                    PyObject* pItem = PyList_GetItem(pResult, i);
                    // printf("---3\n");
                    output[i] = PyFloat_AsDouble(pItem);
                    //printf("down here4\n");
                    // Note: No need to decrement reference for pItem since PyList_GetItem returns a borrowed reference
                }

            }

            Py_XDECREF(pArgs);
            Py_XDECREF(pResult);
        } else {
            //printf("not callable");
        }
        Py_XDECREF(pMethod);
    } else {
        printf("hitpoint pointer was null\n");
    }

    // Unlock the mutex to allow other threads to access the model
    pthread_mutex_unlock(&model_mutex);
    
    //printf("hello\n");
    
}



void NeuralRayTracer_GetShading(NeuralRayTracer* instance, double origin[3], double az_el[2], double output[3]) {
    
    // THE SEG FAULT COMES FROM RUNNING THE FOLLOWING CODE ON MULTIPLE CPUS AT THE SAME TIME

    // printf("in shoot ray method\n");
    //printf("pModelInstance: %p\n", (void*)instance->pModelInstance);
    // Lock the mutex to ensure exclusive access to the model
    pthread_mutex_lock(&model_mutex);

    //printf("IN NEURAL RAY TRACER SHOOT RAY METHOD in cpu: %s\n", buffer);
    if (instance->pModel2Instance != NULL) {

        //printf("before predict\n");
        // printf("pModelInstance: %p\n", (void*)instance->pModelInstance);
        int present = PyObject_HasAttrString(instance->pModel2Instance, "predict_dist_az_el");

        // printf("in here shootray 1\n");
        PyObject* pMethod = PyObject_GetAttrString(instance->pModel2Instance, "predict_dist_az_el");
        // printf("checking if callable!\n");
        if (PyCallable_Check(pMethod)) {
            // printf("its callable 2\n");

            PyObject* pArgs = PyTuple_Pack(2, 
                                           Py_BuildValue("[d,d,d]", origin[0], origin[1], origin[2]),
                                           Py_BuildValue("[d,d]", az_el[0], az_el[1]));

            // printf("about to call!\n");
            
            PyObject* pResult = PyObject_CallObject(pMethod, pArgs);

            // printf("called\n");

            
            if (PyList_Check(pResult) && PyList_Size(pResult) == 3) {
                //double values[6];
                //printf("down here\n")
                // printf("---1\n");;
                for (int i = 0; i < 3; i++) {
                    //printf("down here2\n");
                    // printf("---2\n");
                    PyObject* pItem = PyList_GetItem(pResult, i);
                    // printf("---3\n");
                    output[i] = PyFloat_AsDouble(pItem);
                    //printf("down here4\n");
                    // Note: No need to decrement reference for pItem since PyList_GetItem returns a borrowed reference
                }

            }

            Py_XDECREF(pArgs);
            Py_XDECREF(pResult);
        } else {
            //printf("not callable");
        }
        Py_XDECREF(pMethod);
    } else {
        printf("shading pointer was null\n");
    }

    // Unlock the mutex to allow other threads to access the model
    pthread_mutex_unlock(&model_mutex);
    
}

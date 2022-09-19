#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <stdbool.h>

/*
 * numpy ufuncs used in the coordinates module for performance
 */

static PyMethodDef optimizationMethods[] = {
    {NULL, NULL, 0, NULL}
};

static void double_wrap_at(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *angle_ptr = args[0];
    char *wrap_angle_ptr = args[1];
    char *full_circle_ptr = args[2];
	char *out = args[3];

    double angle;
	double wrap_angle;
	double wrap_angle_floor;
	double full_circle;
	int n_wraps;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
		angle = *(double *) angle_ptr;

		if (isfinite(angle)) {
			wrap_angle  = *(double *) wrap_angle_ptr;
			full_circle = *(double *) full_circle_ptr;
			wrap_angle_floor = wrap_angle - full_circle;
			n_wraps = floor((angle - wrap_angle_floor) / full_circle);
			if (n_wraps != 0) {
				angle -= n_wraps * full_circle;
			}

			if (angle >= wrap_angle) {
				angle -= full_circle;
			}
			if (angle < wrap_angle_floor) {
				angle += full_circle;
			}
		}

        *((double *)out) = angle;
        /* END main ufunc computation */

		angle_ptr += steps[0];
		wrap_angle_ptr += steps[1];
		full_circle_ptr += steps[2];
		out += steps[3];
    }
}


static void float_wrap_at(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *angle_ptr = args[0];
    char *wrap_angle_ptr = args[1];
    char *full_circle_ptr = args[2];
	char *out = args[3];

    float angle;
	float wrap_angle;
	float wrap_angle_floor;
	float full_circle;
	int n_wraps;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
		angle = *(float *) angle_ptr;

		if (isfinite(angle)) {
			wrap_angle  = *(float *) wrap_angle_ptr;
			full_circle = *(float *) full_circle_ptr;
			wrap_angle_floor = wrap_angle - full_circle;
			n_wraps = floor((angle - wrap_angle_floor) / full_circle);
			if (n_wraps != 0) {
				angle -= n_wraps * full_circle;
			}

			if (angle >= wrap_angle) {
				angle -= full_circle;
			}
			if (angle < wrap_angle_floor) {
				angle += full_circle;
			}
		}

        *((float *)out) = angle;
        /* END main ufunc computation */

		angle_ptr += steps[0];
		wrap_angle_ptr += steps[1];
		full_circle_ptr += steps[2];
		out += steps[3];
    }
}


static void int32_wrap_at(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *angle_ptr = args[0];
    char *wrap_angle_ptr = args[1];
    char *full_circle_ptr = args[2];
	char *out = args[3];

    npy_int32 angle;
	npy_int32 wrap_angle;
	npy_int32 wrap_angle_floor;
	npy_int32 full_circle;
	int n_wraps;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
		angle = *(npy_int32 *) angle_ptr;

		wrap_angle  = *(npy_int32 *) wrap_angle_ptr;
		full_circle = *(npy_int32 *) full_circle_ptr;
		wrap_angle_floor = wrap_angle - full_circle;

		n_wraps = floor((angle - wrap_angle_floor) / full_circle);
		if (n_wraps != 0) {
			angle -= n_wraps * full_circle;
		}
		if (angle >= wrap_angle) {
			angle -= full_circle;
		}
		if (angle < wrap_angle_floor) {
			angle += full_circle;
		}

        *((npy_int32 *)out) = angle;
        /* END main ufunc computation */

		angle_ptr += steps[0];
		wrap_angle_ptr += steps[1];
		full_circle_ptr += steps[2];
		out += steps[3];
    }
}


static void int64_wrap_at(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *angle_ptr = args[0];
    char *wrap_angle_ptr = args[1];
    char *full_circle_ptr = args[2];
	char *out = args[3];

    npy_int64 angle;
	npy_int64 wrap_angle;
	npy_int64 wrap_angle_floor;
	npy_int64 full_circle;
	int n_wraps;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
		angle = *(npy_int64 *) angle_ptr;

		wrap_angle  = *(npy_int64 *) wrap_angle_ptr;
		full_circle = *(npy_int64 *) full_circle_ptr;
		wrap_angle_floor = wrap_angle - full_circle;

		n_wraps = floor((angle - wrap_angle_floor) / full_circle);
		if (n_wraps != 0) {
			angle -= n_wraps * full_circle;
		}
		if (angle >= wrap_angle) {
			angle -= full_circle;
		}
		if (angle < wrap_angle_floor) {
			angle += full_circle;
		}

        *((npy_int64 *)out) = angle;
        /* END main ufunc computation */

		angle_ptr += steps[0];
		wrap_angle_ptr += steps[1];
		full_circle_ptr += steps[2];
		out += steps[3];
    }
}


/* The loop definitions must precede the PyMODINIT_FUNC. */



/*This gives pointers to the above functions*/
PyUFuncGenericFunction wrap_at_funcs[] = {
    &int32_wrap_at,
    &int64_wrap_at,
    &float_wrap_at,
    &double_wrap_at
};

static char wrap_at_types[] = {
    NPY_INT32, NPY_INT32, NPY_INT32, NPY_INT32,
    NPY_INT64, NPY_INT64, NPY_INT64, NPY_INT64,
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE
};



static void* data[] = {NULL, NULL, NULL, NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_optimizations",
    NULL,
    -1,
    optimizationMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__optimizations(void)
{
    PyObject *m, *wrap_at, *needs_wrapping, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    wrap_at = PyUFunc_FromFuncAndData(
		wrap_at_funcs,
		data,
		wrap_at_types,
		sizeof(wrap_at_funcs) / sizeof(PyUFuncGenericFunction),
		3 /* input arguments */,
		1 /* output arguments */,
		PyUFunc_None,
		"_wrap_at",
		"wrap_at_docstring",
		0
	);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "_wrap_at", wrap_at);
    Py_DECREF(wrap_at);

    return m;
}

/*                        W O R K E R . C
 * BRL-CAD
 *
 * Copyright (c) 1985-2023 United States Government as represented by
 * the U.S. Army Research Laboratory.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * version 2.1 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this file; see the file named COPYING for more
 * information.
 */
/** @file rt/worker.c
 *
 * Routines to handle initialization of the grid, and dispatch of the
 * first rays from the eye.
 *
 */

#include "common.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bu/log.h"
#include "vmath.h"
#include <math.h>
#include "bn.h"
#include "raytrace.h"
#include "dm.h"		/* Added because RGBpixel is now needed in do_pixel() */

#include "./rtuif.h"
#include "./ext.h"
#include "NeuralRayTracer.h"
#include <stdio.h>
#include <math.h>


/* for fork/pipe linux timing hack */
#ifdef USE_FORKED_THREADS
#  include <sys/select.h>
#  include <sys/types.h>
#  ifdef HAVE_SYS_WAIT_H
#    include <sys/wait.h>
#  endif
#endif

#define PI 3.14159265358979323846
#define CRT_BLEND(v)	(0.26*(v)[X] + 0.66*(v)[Y] + 0.08*(v)[Z])
#define NTSC_BLEND(v)	(0.30*(v)[X] + 0.59*(v)[Y] + 0.11*(v)[Z])

extern fastf_t** timeTable_init(int x, int y);
extern int timeTable_input(int x, int y, fastf_t t, fastf_t **timeTable);

extern int query_x;
extern int query_y;
extern int Query_one_pixel;
extern int query_optical_debug;
extern int query_debug;

extern unsigned char *pixmap;	/* pixmap for rerendering of black pixels */

int per_processor_chunk = 0;	/* how many pixels to do at once */

int fullfloat_mode = 0;
int reproject_mode = 0;
struct floatpixel *curr_float_frame; /* buffer of full frame */
struct floatpixel *prev_float_frame;
int reproj_cur;	/* number of pixels reprojected this frame */
int reproj_max;	/* out of total number of pixels */

/* Local communication with worker() */
int cur_pixel = 0;			/* current pixel number, 0..last_pixel */
int last_pixel = 0;			/* last pixel number */

int stop_worker = 0;

const char *database_name;
NeuralRayTracer * global_nrt;
int training_flag = -1;
struct application * global_ap;

#define ARRAY_SIZE 6

typedef struct Node {
    double arr[ARRAY_SIZE];
    struct Node* next;
} Node;

// Create a new node and return its pointer
Node* createNode(double values[ARRAY_SIZE]) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        perror("Failed to create new node");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < ARRAY_SIZE; i++) {
        newNode->arr[i] = values[i];
    }
    newNode->next = NULL;

    return newNode;
}

// Convert degrees to radians
double degrees_to_radians(double degrees) {
    return degrees * (PI / 180.0);
}

// Normalize azimuth to be within [0, 2π] radians
double normalize_azimuth(double azimuth) {
    azimuth = fmod(azimuth, 360.0); // Modulo to keep within [0, 360]
    if (azimuth < 0) {
        azimuth += 360.0; // Fix negative values
    }

    if(azimuth < 0 || azimuth > 360) {
        printf("AZIMUTH IS OUT OF BOUNDS\n");
    }
    return degrees_to_radians(azimuth);
}

// Normalize elevation to be within [-π/2, π/2] radians
double normalize_elevation(double elevation) {

    // if(elevation < -90) {
    //     elevation = -90;
    // } else if (elevation > 90) {
    //     elevation = 90;
    // }
    // Assuming elevation is always provided within [-90, 90]
    if(elevation < -90) {
        elevation = -90.0;
    } else if (elevation > 90.0) {
        elevation = 90.0;
    }
    return degrees_to_radians(elevation);
}

// Add a node to the end of the linked list
void appendNode(Node** head, double values[ARRAY_SIZE]) {
    Node* newNode = createNode(values);
    if (*head == NULL) {
        *head = newNode;
        return;
    }

    Node* temp = *head;
    while (temp->next != NULL) {
        temp = temp->next;
    }
    temp->next = newNode;
}

/**
 * For certain hypersample values there is a particular advantage to
 * subdividing the pixel and shooting a ray in each sub-pixel.  This
 * structure keeps track of those patterns
 */
struct jitter_pattern {
    int num_samples;/* number of samples, or coordinate pairs in coords[] */
    double rand_scale[2]; /* amount to scale bn_rand_half value */
    double coords[32]; /* center of each sub-pixel */
};


static struct jitter_pattern pt_pats[] = {

    {4, {0.5, 0.5}, 	/* -H 3 */
     { 0.25, 0.25,
       0.25, 0.75,
       0.75, 0.25,
       0.75, 0.75 } },

    {5, {0.4, 0.4}, 	/* -H 4 */
     { 0.2, 0.2,
       0.2, 0.8,
       0.8, 0.2,
       0.8, 0.8,
       0.5, 0.5} },

    {9, {0.3333, 0.3333}, /* -H 8 */
     { 0.17, 0.17,  0.17, 0.5,  0.17, 0.82,
       0.5, 0.17,    0.5, 0.5,   0.5, 0.82,
       0.82, 0.17,  0.82, 0.5,  0.82, 0.82 } },

    {16, {0.25, 0.25}, 	/* -H 15 */
     { 0.125, 0.125,  0.125, 0.375, 0.125, 0.625, 0.125, 0.875,
       0.375, 0.125,  0.375, 0.375, 0.375, 0.625, 0.375, 0.875,
       0.625, 0.125,  0.625, 0.375, 0.625, 0.625, 0.625, 0.875,
       0.875, 0.125,  0.875, 0.375, 0.875, 0.625, 0.875, 0.875} },

    { 0, {0.0, 0.0}, {0.0} } /* must be here to stop search */
};


/**
 * Compute the origin for this ray, based upon the number of samples
 * per pixel and the number of the current sample.  For certain
 * ray-counts, it is highly advantageous to subdivide the pixel and
 * fire each ray in a specific sub-section of the pixel.
 */
static void
jitter_start_pnt(vect_t point, struct application *a, int samplenum, int pat_num)
{
    fastf_t dx, dy;

    if (pat_num >= 0) {
	dx = a->a_x + pt_pats[pat_num].coords[samplenum*2] +
	    (bn_rand_half(a->a_resource->re_randptr) *
	     pt_pats[pat_num].rand_scale[X]);

	dy = a->a_y + pt_pats[pat_num].coords[samplenum*2 + 1] +
	    (bn_rand_half(a->a_resource->re_randptr) *
	     pt_pats[pat_num].rand_scale[Y]);
    } else {
	dx = a->a_x + bn_rand_half(a->a_resource->re_randptr);
	dy = a->a_y + bn_rand_half(a->a_resource->re_randptr);
    }
    VJOIN2(point, viewbase_model, dx, dx_model, dy, dy_model);
}


#include <stdio.h>
#include <math.h>

int roundToNearest90_az(int value) {
    // Normalize value to be within [0, 360)
    value = value % 360;
    if (value < 0) {
        value += 360;
    }

    // Determine the nearest multiple of 90
    int quotient = value / 90;
    int remainder = value % 90;

    // Round to the nearest multiple
    if (remainder < 45) {
        value = quotient * 90;
    } else {
        value = (quotient + 1) * 90;
    }

    // Normalize again in case we have rounded to 360
    value = value % 360;

    return value;
}
int roundToNearest90_el(int value) {
    // The value is already one of the possible outcomes
    if (value == 0 || value == 90 || value == -90) {
        return value;
    }

    // Find the distance to -90, 0, and 90
    int distanceToMinus90 = abs(value + 90);
    int distanceToZero = abs(value);
    int distanceTo90 = abs(value - 90);

    // Determine the closest distance
    if (distanceToZero <= distanceTo90 && distanceToZero <= distanceToMinus90) {
        return 0;
    } else if (distanceTo90 < distanceToZero && distanceTo90 <= distanceToMinus90) {
        return 90;
    } else {
        return -90;
    }
}
int roundToNearest60_az(int value) {
    // Normalize value to be within [0, 360)
    value = value % 360;
    if (value < 0) {
        value += 360;
    }

    // Determine the nearest multiple of 60
    int quotient = value / 60;
    int remainder = value % 60;

    // Round to the nearest multiple
    if (remainder < 30) {
        value = quotient * 60;
    } else {
        value = (quotient + 1) * 60;
    }

    // Normalize again in case we have rounded to 360
    value = value % 360;

    return value;
}

int roundToNearest60_el(int value) {
    // The value is already one of the possible outcomes
    if (value == 0 || value == 60 || value == -60) {
        return value;
    }

    // Find the distance to -60, 0, and 60
    int distanceToMinus60 = abs(value + 60);
    int distanceToZero = abs(value);
    int distanceTo60 = abs(value - 60);

    // Determine the closest distance
    if (distanceToZero <= distanceTo60 && distanceToZero <= distanceToMinus60) {
        return 0;
    } else if (distanceTo60 < distanceToZero && distanceTo60 <= distanceToMinus60) {
        return 60;
    } else {
        return -60;
    }
}




void
do_pixel(int cpu, int pat_num, int pixelnum)
{
    
    //printf("in do_pixel\n");
    struct application a;
    struct pixel_ext pe;
    vect_t stereo_point;		/* Ref point on eye or view plane */
    vect_t point;		/* Ref point on eye or view plane */
    vect_t colorsum = {(fastf_t)0.0, (fastf_t)0.0, (fastf_t)0.0};
    int samplenum = 0;
    static const double one_over_255 = 1.0 / 255.0;
    const int pindex = (pixelnum * sizeof(RGBpixel));

    /* for stereo output */
    vect_t left_eye_delta = VINIT_ZERO;

    if (lightmodel == 8) {
	/* Add timer here to start pixel-time for heat
	 * graph, when asked.
	 */
	rt_prep_timer();
    }

    //printf("obtaining app\n");
    /* Obtain fresh copy of global application struct */

    /*
    if(training_flag == 1) {
        a = *global_ap;
    } else {
        
    }
    */

    a = APP;				/* struct copy */
    a.a_resource = &resource[cpu];

    if (incr_mode) {

	register int i = 1<<incr_level;

	a.a_y = pixelnum/i;
	a.a_x = pixelnum - (a.a_y * i);
	/* a.a_x = pixelnum%i; */
	if (incr_level != 0) { // NORMAL RT DOES NOT GO IN HERE

	    /* See if already done last pass */
	    if (((a.a_x & 1) == 0) &&
		((a.a_y & 1) == 0))
		return;
	}
	a.a_x <<= (incr_nlevel-incr_level);
	a.a_y <<= (incr_nlevel-incr_level);
    } else { // NORMAL RT GOES IN HERE

    a.a_y = (int)(pixelnum/width);
    //printf("a.a_y: %d\n", a.a_y);
    //printf("width: %f\n", width);
    //printf("width: %d\n", width);
    a.a_x = (int)(pixelnum - (a.a_y * width));
    //printf("a.a_x: %d\n", a.a_x);
    /* a.a_x = pixelnum%width; */
    }

    
    if (Query_one_pixel) { // NORMAL RT DOES NOT GO IN HERE
	if (a.a_x == query_x && a.a_y == query_y) {
	    optical_debug = query_optical_debug;
	    rt_debug = query_debug;
	} else {
	    rt_debug = optical_debug = 0;
	}
    }

    if (sub_grid_mode) { // NORMAL RT DOES NOT GO IN HERE
	if (a.a_x < sub_xmin || a.a_x > sub_xmax)
	    return;
	if (a.a_y < sub_ymin || a.a_y > sub_ymax)
	    return;
    }

    if (fullfloat_mode) { // NORMAL RT DOES NOT GO IN HERE

	register struct floatpixel *fp;
	fp = &curr_float_frame[a.a_y*width + a.a_x];
	if (fp->ff_frame >= 0) {
	    return;	/* pixel was reprojected */
	}
    }


    /* Check the pixel map to determine if this image should be
     * rendered or not.
     */
    if (pixmap) {

	a.a_user= 1;	/* Force Shot Hit */

	if (pixmap[pindex + RED] + pixmap[pindex + GRN] + pixmap[pindex + BLU]) { // NORMAL RT DOES NOT GO IN HERE
	    /* non-black pixmap pixel */
        //printf("calling view pixel\n");

        // This is NOT where the color is being set
        
	    a.a_color[RED]= (double)(pixmap[pindex + RED]) * one_over_255;
	    a.a_color[GRN]= (double)(pixmap[pindex + GRN]) * one_over_255;
	    a.a_color[BLU]= (double)(pixmap[pindex + BLU]) * one_over_255;

	    /* we're done */
	    view_pixel(&a);
	    if ((size_t)a.a_x == width-1) {
		view_eol(&a);		/* End of scan line */
	    }
	    return;
	}
    }

    /* our starting point, used for non-jitter */
    VJOIN2 (point, viewbase_model, a.a_x, dx_model, a.a_y, dy_model);


    /* not tracing the corners of a prism by default */
    a.a_pixelext=(struct pixel_ext *)NULL;

    /* black or no pixmap, so compute the pixel(s) */

    /* LOOP BELOW IS UNROLLED ONE SAMPLE SINCE THAT'S THE COMMON CASE.
     *
     * XXX - If you edit the unrolled or non-unrolled section, be sure
     * to edit the other section.
     */

    if (hypersample == 0) {
	/* not hypersampling, so just do it */

	/****************/
	/* BEGIN UNROLL */
	/****************/

	if (jitter & JITTER_CELL) {
	    jitter_start_pnt(point, &a, samplenum, pat_num);
	}

    //printf("here1\n");

	if (a.a_rt_i->rti_prismtrace) {
	    /* compute the four corners */
	    pe.magic = PIXEL_EXT_MAGIC;
	    VJOIN2(pe.corner[0].r_pt, viewbase_model, a.a_x, dx_model, a.a_y, dy_model);
	    VJOIN2(pe.corner[1].r_pt, viewbase_model, (a.a_x+1), dx_model, a.a_y, dy_model);
	    VJOIN2(pe.corner[2].r_pt, viewbase_model, (a.a_x+1), dx_model, (a.a_y+1), dy_model);
	    VJOIN2(pe.corner[3].r_pt, viewbase_model, a.a_x, dx_model, (a.a_y+1), dy_model);
	    a.a_pixelext = &pe;
	} 


	if (rt_perspective > 0.0) {
	    VSUB2(a.a_ray.r_dir, point, eye_model);
	    VUNITIZE(a.a_ray.r_dir);
	    VMOVE(a.a_ray.r_pt, eye_model);
	    if (a.a_rt_i->rti_prismtrace) {
		VSUB2(pe.corner[0].r_dir, pe.corner[0].r_pt, eye_model);
		VSUB2(pe.corner[1].r_dir, pe.corner[1].r_pt, eye_model);
		VSUB2(pe.corner[2].r_dir, pe.corner[2].r_pt, eye_model);
		VSUB2(pe.corner[3].r_dir, pe.corner[3].r_pt, eye_model);
	    }
	} else {
        //printf("here 2.5\n");

        //printf("setting this though!\n");
	    VMOVE(a.a_ray.r_pt, point);
	    VMOVE(a.a_ray.r_dir, APP.a_ray.r_dir);

        //printf("point: [%f, %f, %f]\n", a.a_ray.r_pt[0], a.a_ray.r_pt[1], a.a_ray.r_pt[2]);
        //printf("ray dir: [%f, %f, %f]\n", a.a_ray.r_dir[0], a.a_ray.r_dir[1], a.a_ray.r_dir[2]);

	    if (a.a_rt_i->rti_prismtrace) {
		VMOVE(pe.corner[0].r_dir, a.a_ray.r_dir);
		VMOVE(pe.corner[1].r_dir, a.a_ray.r_dir);
		VMOVE(pe.corner[2].r_dir, a.a_ray.r_dir);
		VMOVE(pe.corner[3].r_dir, a.a_ray.r_dir);
	    }
	}

    //printf("here3\n");
	if (report_progress) {
	    report_progress = 0;
	    bu_log("\tframe %d, xy=%d, %d on cpu %d, samp=%d\n", curframe, a.a_x, a.a_y, cpu, samplenum);
	}
 
	a.a_level = 0;		/* recursion level */
	a.a_purpose = "main ray";
    //(void)rt_shootray(&a); // This is the call to rt_shootray we need to edit

    //printf("here4\n");
    // Traditional raytrace
    if(neural_rendering != 1) {


        // FILE *file = fopen("real_ray_origins.txt", "a");
        
        // if (file == NULL) {
        //     printf("Error opening the file.\n");
        //     exit(0);
        // }

        // convert direction to azimuth and elevation
        fastf_t azp;
        fastf_t elp;

        bn_ae_vec(&azp, &elp, a.a_ray.r_dir);

        fastf_t azp2 = azp / 36.0;
        fastf_t elp2 = elp / 9.0;

        // fprintf(file, "%lf, %lf, %lf, %lf, %lf\n", a.a_ray.r_pt[0], a.a_ray.r_pt[1], a.a_ray.r_pt[2], azp2, elp2);

        // fclose(file);

        //printf("doing neural rendering!\n");
        (void)rt_shootray(&a); // This is the call to rt_shootray we need to edit


        if(a.a_return != 0) {
            /*
            printf("ray hit!\n");
            printf("a_color[0]: %f\n", a.a_color[0]);
            printf("a_color[1]: %f\n", a.a_color[1]);
            printf("a_color[2]: %f\n", a.a_color[2]);
            printf("colorsum[0]: %f\n", colorsum[0]);
            printf("colorsum[1]: %f\n", colorsum[1]);
            printf("colorsum[2]: %f\n", colorsum[2]);
            */
        }
    }
    else { 

        // Open or create a file called real_ray_origins in append mode
        
        // FILE *file = fopen("real_ray_origins.txt", "a");
        
        // if (file == NULL) {
        //     printf("Error opening the file.\n");
        //     exit(0);
        // }

        // fprintf(file, "%lf, %lf, %lf, %lf, %lf, %lf\n", a.a_ray.r_pt[0], a.a_ray.r_pt[1], a.a_ray.r_pt[2], a.a_ray.r_dir[0], a.a_ray.r_dir[1], a.a_ray.r_dir[2]);

        // fclose(file);
        


        // printf("doing neural rendering!\n");

        double ray_origin[3];
        double ray_dir[3];
        double model_output[1];
 
        for(int i = 0; i < 3; i++) {
            ray_origin[i] = (double) a.a_ray.r_pt[i];
            ray_dir[i] = (double) a.a_ray.r_dir[i];
            
        }

        // printf("ray origin: [%f, %f, %f]\n",ray_origin[0], ray_origin[1], ray_origin[2]);

        
        // convert direction to azimuth and elevation
        fastf_t azp;
        fastf_t elp;

        bn_ae_vec(&azp, &elp, a.a_ray.r_dir);

        fastf_t stored_az = azp;
        fastf_t stored_el = elp;

        // printf("az: %f el: %f", azp, elp);

        // printf("NR2!\n");

        double az_el_vec[2];
        az_el_vec[0] = (double) azp / 36.0;
        az_el_vec[1] = (double) elp / 9.0;
        // double epsilon = 0.1;

        // printf("NR3!\n");


        // Need to go from dir to -a -e

        //printf("Model input: ray origin: [%f, %f, %f, %f, %f]\n", ray_origin[0], ray_origin[1], ray_origin[2], az_el_vec[0], az_el_vec[1]);

        //printf("global nrt: \n", global_nrt);

        NeuralRayTracer_ShootRay(global_nrt, ray_origin, az_el_vec, model_output);

        // Now you can print the values
        // printf("Predicted values / model output: [%f]\n", model_output[0]);

        /*
        double dist = model_output[0];
        
        */

        double hit_or_miss = model_output[0];
        // double returned_az = model_output[1] * 36.0;
        // double returned_el = model_output[2] * 9.0;

        //printf("hit or miss! %f\n: ", hit_or_miss);

        //double dist = .1;

        // If distance == -1 then it was a miss
        if (hit_or_miss < .94) {
           //a.a_return = 0; 
           a.a_return = 0;
           a.a_miss(&a);
           //printf("ray missed!\n");
           a.a_color[0] = 0;
           a.a_color[1] = 0;
           a.a_color[2] = 0;
        //    printf("ray miss!!!\n");
        } else {
            // printf("ray hit!\n");
            // This is what sets the color!
            a.a_color[0] = 0.5;
            a.a_color[1] = 0.5;
            a.a_color[2] = 0.5;

            /*

            double dist_az_el[3];

            NeuralRayTracer_GetShading(global_nrt, ray_origin, az_el_vec, dist_az_el);

            // printf("Predicted values / model output: [%f, %f, %f].\n", dist_az_el[0], dist_az_el[1], dist_az_el[2]);

            double normal_vector_az = (dist_az_el[1] * 36.0) + 180;
            double normal_vector_el = dist_az_el[2] * 9.0;

            normal_vector_az = roundToNearest90_az(normal_vector_az);
            normal_vector_el = roundToNearest90_el(normal_vector_el);

            // printf("normal vector az: %f\n", normal_vector_az);
            // printf("normal vector el: %f\n", normal_vector_el);
            


            double point_az_rad = normalize_azimuth(normal_vector_az);
            double point_el_rad = normalize_elevation(normal_vector_el);
            double light_az_rad = normalize_azimuth((double)azp);
            double light_el_rad = normalize_elevation((double)elp * -1);


            // printf("azp: %f\n", azp);
            // printf("elp: %f\n", elp);
            
            
            // printf("point_az_rad: %f\n", point_az_rad);
            // printf("point_el_rad: %f\n", point_el_rad);
            // printf("light_az_rad: %f\n", light_az_rad);
            // printf("light_el_rad: %f\n", light_el_rad);
            // Convert spherical coordinates to Cartesian coordinates for the normal vector
            double nx = cos(point_el_rad) * cos(point_az_rad);
            double ny = cos(point_el_rad) * sin(point_az_rad);
            double nz = sin(point_el_rad);

            // Convert spherical coordinates to Cartesian coordinates for the light direction
            double lx = cos(light_el_rad) * cos(light_az_rad);
            double ly = cos(light_el_rad) * sin(light_az_rad);
            double lz = sin(light_el_rad);

            // printf("Normal Vector (nx, ny, nz): (%f, %f, %f)\n", nx, ny, nz);
            // printf("Light Vector (lx, ly, lz): (%f, %f, %f)\n", lx, ly, lz);


            // Dot product between normal and light direction vectors
            double intensity = nx * lx + ny * ly + nz * lz;
            // printf("intensity: %f\n", intensity);
            // intensity = abs(intensity);

            // Clamp intensity to the [0, 1] range
            intensity = fmax(0, intensity); // No negative values
            intensity = fmin(1, intensity); // No values greater than 1

            intensity = round(intensity * 10.0) / 10.0;

            a.a_color[0] = intensity; // Red component
            a.a_color[1] = intensity; // Green component
            a.a_color[2] = intensity; // Blue component
            // //printf("ray hit!\n");
            */
        }
    }

   
    //printf("in flag loop\n");
    VADD2(colorsum, colorsum, a.a_color); // If i comment this out w/ normal rt what happens
    /**************/
    /* END UNROLL */
    /**************/

    

    /* bu_log("2: [%d, %d] : [%.2f, %.2f, %.2f]\n", pixelnum%width, pixelnum/width, a.a_color[0], a.a_color[1], a.a_color[2]); */

    /* Add get_pixel_timer here to get total time taken to get pixel, when asked */
    if (lightmodel == 8) {
    fastf_t pixelTime;
    fastf_t **timeTable;

    pixelTime = rt_get_timer(NULL,NULL); /* FIXME: needs to use bu_gettime() */
    /* bu_log("PixelTime = %lf X:%d Y:%d\n", pixelTime, a.a_x, a.a_y); */
    bu_semaphore_acquire(RT_SEM_RESULTS);
    timeTable = timeTable_init(width, height);
    timeTable_input(a.a_x, a.a_y, pixelTime, timeTable);
    bu_semaphore_release(RT_SEM_RESULTS);
    }

    /* we're done */
    view_pixel(&a); // colorsum comes into play here
    if ((size_t)a.a_x == width-1) {
    view_eol(&a);		/* End of scan line */
    }

} // hypersample check

    //printf("returning from do_pixel\n");
    
	
    return;
}



/**
 * Compute some pixels, and store them.
 *
 * This uses a "self-dispatching" parallel algorithm.  Executes until
 * there is no more work to be done, or is told to stop.
 *
 * In order to reduce the traffic through the res_worker critical
 * section, a multiple pixel block may be removed from the work queue
 * at once.
 */
void
worker(int cpu, void *UNUSED(arg))
{
    
    // printf("in worker\n");
    int pixel_start;
    int pixelnum;
    int pat_num = -1;

    char buffer[20];  // Ensure the buffer is large enough to store the number


    sprintf(buffer, "%d", cpu);
    //printf("in worker with cpu: %s\n", buffer);
    // Create tracer instance for this cpu
    // NeuralRayTracer * tracer_instance = NeuralRayTracer_Create(model_path);

    /* Figure out a reasonable chunk size that should keep most
     * workers busy all the way to the end.  We divide up the image
     * into chunks equating to tiles 1x1, 2x2, 4x4, 8x8 ... in size.
     * Work is distributed so that all CPUs work on at least 8 chunks
     * with the chunking adjusted from a maximum chunk size (512x512)
     * all the way down to 1 pixel at a time, depending on the number
     * of cores and the size of our rendering.
     *
     * TODO: actually work on image tiles instead of pixel spans.
     */
    if (per_processor_chunk <= 0) {
	size_t chunk_size;
	size_t one_eighth = (last_pixel - cur_pixel) * (hypersample + 1) / 8;
	if (UNLIKELY(one_eighth < 1))
	    one_eighth = 1;

	if (one_eighth > (size_t)npsw * 262144)
	    chunk_size = 262144; /* 512x512 */
	else if (one_eighth > (size_t)npsw * 65536)
	    chunk_size = 65536; /* 256x256 */
	else if (one_eighth > (size_t)npsw * 16384)
	    chunk_size = 16384; /* 128x128 */
	else if (one_eighth > (size_t)npsw * 4096)
	    chunk_size = 4096; /* 64x64 */
	else if (one_eighth > (size_t)npsw * 1024)
	    chunk_size = 1024; /* 32x32 */
	else if (one_eighth > (size_t)npsw * 256)
	    chunk_size = 256; /* 16x16 */
	else if (one_eighth > (size_t)npsw * 64)
	    chunk_size = 64; /* 8x8 */
	else if (one_eighth > (size_t)npsw * 16)
	    chunk_size = 16; /* 4x4 */
	else if (one_eighth > (size_t)npsw * 4)
	    chunk_size = 4; /* 2x2 */
	else
	    chunk_size = 1; /* one pixel at a time */

    //printf("in here\n");

	bu_semaphore_acquire(RT_SEM_WORKER);
	per_processor_chunk = chunk_size;
	bu_semaphore_release(RT_SEM_WORKER);

    //printf("in here2\n");
    }

    //printf("down here now\n");

    if (cpu >= MAX_PSW) {
	bu_log("rt/worker() cpu %d > MAX_PSW %d, array overrun\n", cpu, MAX_PSW);
	bu_exit(EXIT_FAILURE, "rt/worker() cpu > MAX_PSW, array overrun\n");
    }


    RT_CK_RESOURCE(&resource[cpu]);


    
    pat_num = -1;
    if (hypersample) {
	int i, ray_samples;

	ray_samples = hypersample + 1;
	for (i=0; pt_pats[i].num_samples != 0; i++) {
	    if (pt_pats[i].num_samples == ray_samples) {
		pat_num = i;
		goto pat_found;
	    }
	}
    }

pat_found:

    if (random_mode) {

	/* FIXME: this currently runs forever. It should probably
	 *        generate a list of random pixels and then process
	 *        them in that order.
	 */

	while (1) {
	    /* Generate a random pixel id between 0 and last_pixel
	       inclusive - TODO: check if there is any issue related
	       with multi-threaded RNG */
	    pixelnum = rand()*1.0/RAND_MAX*(last_pixel + 1);
	    if (pixelnum >= last_pixel)
		pixelnum = last_pixel;
	    do_pixel(cpu, pat_num, pixelnum);
	}

    } else {
        int from;
        int to;

        //printf("not random mode\n");

        while (1) {
            if (stop_worker)
            return;
            
            bu_semaphore_acquire(RT_SEM_WORKER);
            pixel_start = cur_pixel;
            cur_pixel += per_processor_chunk;
            bu_semaphore_release(RT_SEM_WORKER);
            

            if (top_down) {
            from = last_pixel - pixel_start;
            to = from - per_processor_chunk;
            } else {
            from = pixel_start;
            to = pixel_start + per_processor_chunk;
            }

            //printf("down here now\n");
            /* bu_log("SPAN[%d -> %d] for %d pixels\n", pixel_start, pixel_start+per_processor_chunk, per_processor_chunk); */
            for (pixelnum = from; pixelnum != to; (from < to) ? pixelnum++ : pixelnum--) {
            if (pixelnum > last_pixel || pixelnum < 0)
                return;

            //printf("down here now2\n");
            /* bu_log("    PIXEL[%d]\n", pixelnum); */
            do_pixel(cpu, pat_num, pixelnum);

            //printf("back from do_pixel\n");
            }
        }
    }
    //NeuralRayTracer_Destroy(tracer_instance);
}


/**
 * Compute a run of pixels, in parallel if the hardware permits it.
 nrt_instance will be NULL if not doing traditional ray tracing (i.e doing neural rendering)
 */
void
do_run(int a, int b, const char* db_name, NeuralRayTracer * nrt, int neural_training, struct application *ap_temp)
{
    //printf("in do run\n");

    //printf("pixel a: %d\n", a);
    //printf("pixel b: %d\n", b);

    if(neural_training == 1) {
        training_flag = 1;
        global_ap = ap_temp;
    }

    
    // Set global variables for worker
    database_name = db_name;
    //printf("in do run!\n");
    //printf("%s", nrt);
    //printf("\n");
    
    global_nrt = nrt;
    //model_path = model_path_str;

    cur_pixel = a;
    last_pixel = b;

    if (!RTG.rtg_parallel) {
	/*
	 * SERIAL case -- one CPU does all the work.
	 */
	npsw = 1;
	worker(0, NULL);
    } else {
	/*
	 * Parallel case.
	 */
	bu_parallel(worker, (size_t)npsw, NULL);
    }

    //printf("returned from worker\n");

    /* Tally up the statistics */
    size_t cpu;
    for (cpu = 0; cpu < MAX_PSW; cpu++) {
	if (resource[cpu].re_magic != RESOURCE_MAGIC) {
	    bu_log("ERROR: CPU %zu resources corrupted, statistics bad\n", cpu);
	    continue;
	}
	rt_add_res_stats(APP.a_rt_i, &resource[cpu]);
	rt_zero_res_stats(&resource[cpu]);
    }

    return;
}


/*
 * Local Variables:
 * mode: C
 * tab-width: 8
 * indent-tabs-mode: t
 * c-file-style: "stroustrup"
 * End:
 * ex: shiftwidth=4 tabstop=8
 */

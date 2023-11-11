/*                          M A I N . C
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
/** @file rt/main.c
 *
 * Ray Tracing User Interface (RTUIF) main program, using LIBRT
 * library.
 *
 * Invoked by MGED for quick pictures.
 * Is linked with each of several "back ends":
 *	view.c, viewpp.c, viewray.c, viewcheck.c, etc.
 * to produce different executable programs:
 *	rt, rtpp, rtray, rtcheck, etc.
 *
 */

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <signal.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>



#ifdef MPI_ENABLED
#  include <mpi.h>
#endif

#include "bio.h"

#include "bu/app.h"
#include "bu/endian.h"
#include "bu/getopt.h"
#include "bu/bitv.h"
#include "bu/debug.h"
#include "bu/malloc.h"
#include "bu/log.h"
#include "bu/parallel.h"
#include "bu/ptbl.h"
#include "bu/vls.h"
#include "bu/version.h"
#include "vmath.h"
#include "raytrace.h"
#include "dm.h"
#include "pkg.h"

/* private */
#include "./rtuif.h"
#include "./ext.h"
#include "brlcad_ident.h"


extern void application_init(void);

extern const char title[];

/***** Variables shared with viewing model *** */
struct fb	*fbp = FB_NULL;	/* Framebuffer handle */
FILE		*outfp = NULL;		/* optional pixel output file */
struct icv_image *bif = NULL;

/***** end of sharing with viewing model *****/


/***** variables shared with worker() ******/
//struct application APP;
int		report_progress;	/* !0 = user wants progress report */
extern int	incr_mode;		/* !0 for incremental resolution */
extern size_t	incr_nlevel;		/* number of levels */
/***** end variables shared with worker() *****/


/***** variables shared with do.c *****/
extern int	pix_start;		/* pixel to start at */
extern int	pix_end;		/* pixel to end at */
size_t		n_malloc;		/* Totals at last check */
size_t		n_free;
size_t		n_realloc;
extern int	matflag;		/* read matrix from stdin */
extern int	orientflag;		/* 1 means orientation has been set */
extern int	desiredframe;		/* frame to start at */
extern int	curframe;		/* current frame number,
					 * also shared with view.c */
extern char	*outputfile;		/* name of base of output file */
/***** end variables shared with do.c *****/


extern fastf_t	rt_dist_tol;		/* Value for rti_tol.dist */
extern fastf_t	rt_perp_tol;		/* Value for rti_tol.perp */
extern char	*framebuffer;		/* desired framebuffer */

extern struct command_tab rt_do_tab[];


void
siginfo_handler(int UNUSED(arg))
{
    report_progress = 1;
#ifdef SIGUSR1
    (void)signal(SIGUSR1, siginfo_handler);
#endif
#ifdef SIGINFO
    (void)signal(SIGINFO, siginfo_handler);
#endif
}


void
memory_summary(void)
{
    if (rt_verbosity & VERBOSE_STATS) {
	size_t mdelta = bu_n_malloc - n_malloc;
	size_t fdelta = bu_n_free - n_free;
	bu_log("Additional #malloc=%zu, #free=%zu, #realloc=%zu (%zu retained)\n",
	       mdelta,
	       fdelta,
	       bu_n_realloc - n_realloc,
	       mdelta - fdelta);
    }
    n_malloc = bu_n_malloc;
    n_free = bu_n_free;
    n_realloc = bu_n_realloc;
}


int fb_setup(void) {
    /* Framebuffer is desired */
    size_t xx, yy;
    int zoom;

    /* make sure width/height are set via -g/-G */
    grid_sync_dimensions(viewsize);

    /* Ask for a fb big enough to hold the image, at least 512. */
    /* This is so MGED-invoked "postage stamps" get zoomed up big
     * enough to see.
     */
    xx = yy = 512;
    if (xx < width || yy < height) {
	xx = width;
	yy = height;
    }

    bu_semaphore_acquire(BU_SEM_SYSCALL);
    fbp = fb_open(framebuffer, xx, yy);
    bu_semaphore_release(BU_SEM_SYSCALL);
    if (fbp == FB_NULL) {
	fprintf(stderr, "rt:  can't open frame buffer\n");
	return 12;
    }

    bu_semaphore_acquire(BU_SEM_SYSCALL);
    /* If fb came out smaller than requested, do less work */
    size_t fbwidth = (size_t)fb_getwidth(fbp);
    size_t fbheight = (size_t)fb_getheight(fbp);
    if (width > fbwidth)
	width = fbwidth;
    if (height > fbheight)
	height = fbheight;

    /* If fb is lots bigger (>= 2X), zoom up & center */
    if (width > 0 && height > 0) {
	zoom = fbwidth/width;
	if (fbheight/height < (size_t)zoom) {
	    zoom = fb_getheight(fbp)/height;
	}
	(void)fb_view(fbp, width/2, height/2, zoom, zoom);
    }
    bu_semaphore_release(BU_SEM_SYSCALL);

#ifdef USE_OPENCL
    clt_connect_fb(fbp);
#endif
    return 0;
}


static void
initialize_resources(size_t cnt, struct resource *resp, struct rt_i *rtip)
{
    if (!resp)
	return;

    /* Initialize all the per-CPU memory resources.  Number of
     * processors can change at runtime, so initialize all.
     */
    memset(resp, 0, sizeof(struct resource) * cnt);

    int i;
    for (i = 0; i < MAX_PSW; i++) {
	rt_init_resource(&resp[i], i, rtip);
    }
}


static void
initialize_option_defaults(void)
{
    /* GIFT defaults */
    azimuth = 35.0;
    elevation = 25.0;

    /* 40% ambient light */
    AmbientIntensity = 0.4;

    /* 0/0/1 background */
    background[0] = background[1] = 0.0;
    background[2] = 1.0/255.0; /* slightly non-black */

    /* Before option processing, get default number of processors */
    npsw = bu_avail_cpus();		/* Use all that are present */
    if (npsw > MAX_PSW)
	npsw = MAX_PSW;

}


#ifdef MPI_ENABLED
/* MPI atexit() handler */
static void mpi_exit_func(void)
{
    MPI_Finalize();
}
#endif

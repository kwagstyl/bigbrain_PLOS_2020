#! /usr/bin/env perl
#
# Flip y-z coordinates of a slice for more efficient
# editing in register/Display.
#

use strict;
use warnings "all";
use File::Temp qw/ tempdir /;

$ENV{'MINC_FORCE_V2'} = 1;
$ENV{'MINC_COMPRESS'} = 4;

# make tmpdir
my $tmpdir = &tempdir( "flipxy-XXXXXX", TMPDIR => 1, CLEANUP => 1 );


my $input = shift;

if( !( -e $input ) ) {
  die "Slice $input does not exist.";
}

my $dx = `mincinfo -attvalue xspace:step $input`; chomp($dx); $dx += 0.0;
my $dy = `mincinfo -attvalue yspace:step $input`; chomp($dy); $dy += 0.0;
my $dz = `mincinfo -attvalue zspace:step $input`; chomp($dz); $dz += 0.0;
my $nx = `mincinfo -attvalue xspace:length $input`; chomp($nx); $nx += 0;
my $ny = `mincinfo -attvalue yspace:length $input`; chomp($ny); $ny += 0;
my $nz = `mincinfo -attvalue zspace:length $input`; chomp($nz); $nz += 0;
my $sx = `mincinfo -attvalue xspace:start $input`; chomp($sx); $sx += 0.0;
my $sy = `mincinfo -attvalue yspace:start $input`; chomp($sy); $sy += 0.0;
my $sz = `mincinfo -attvalue zspace:start $input`; chomp($sz); $sz += 0.0;

my $flipxfm = "${tmpdir}/flipxy.xfm";
open PIPE, "> ${flipxfm}";
print PIPE "MNI Transform File\n";
print PIPE "Transform_Type = Linear;\n";
print PIPE "Linear_Transform =\n";
print PIPE " 0 1 0 0;\n";
print PIPE " 1 0 0 0\n";
print PIPE " 0 0 1 0\n";
close PIPE;

my $in = "${tmpdir}/flip_in.mnc";
my $out = "${tmpdir}/flip_out.mnc";

&run( 'cp', '-f', $input, $in );

&run( 'mincresample', '-quiet', '-clob', '-unsigned', '-short',
      '-nearest', '-use_input_sampling', '-transform', $flipxfm,
      '-start', $sy, $sx, $sz, '-nelements', $ny, $nx, $nz,
      '-step', $dy, $dx, $dz, '-range', 0, 65535, '-keep_real_range', $in, $out );
&run( 'mv', '-f', $out, $input );

unlink( $flipxfm );


sub run {
   print "@_\n";
   system(@_) == 0 or die;
}


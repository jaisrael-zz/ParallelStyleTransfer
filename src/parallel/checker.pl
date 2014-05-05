#!/usr/bin/perl

use POSIX;

my @scene_names = ("rgb", "rand10k", "rand100k", "pattern", "snowsingle");
#my @scene_names = ("rgb");


my %fast_times = (
        "rgb"       => 0.35 ,
        "rand10k"   => 10.85 , 
        "rand100k"  => 110.80, 
        "pattern"   => 0.88 , 
        "snowsingle"=> 49.67
    );

my %naive_times = (
        "rgb"       => 4.5,
        "rand10k"   => 230 , 
        "rand100k"  => 2305, 
        "pattern"   => 27, 
        "snowsingle"=> 2277 
    );

my $perf_points = 11;
my $correctness_points = 2;

my %correct;

my %time_required;

`mkdir -p logs`;
`rm -rf logs/*`;


print "\n";
print ("--------------\n");
print ("Running tests:\n");
print ("--------------\n");

foreach my $scene (@scene_names) {
    print ("\nScene : $scene\n");
    my @sys_stdout = system ("./render -c $scene -s 768 > ./logs/correctness_${scene}.log");
    my $return_value  = $?;
    if ($return_value == 0) {
        print ("Correctness passed!\n");
        $correct{$scene} = 1;
        

    }
    else {
        print ("Correctness failed ... Check ./logs/correctness_${scene}.log\n");
        $correct{$scene} = 0;
    }


    my $total_time = `./render -r cuda -b 0:4 $scene -s 768 | tee ./logs/time_${scene}.log | grep Total:`;
    chomp($total_time);
    $total_time =~ s/^[^0-9]*//;
    $total_time =~ s/ ms.*//;

    print ("Time required : $total_time\n");
    $time_required{$scene} = $total_time;

}

print "\n";
print ("------------\n");
print ("Score table:\n");
print ("------------\n");

my $header = sprintf ("| %-15s | %-15s | %-15s | %-15s | %-15s |\n", "Scene Name", "Naive Time (Tn)", "Fast Time (To)", "Your Time (T)", "Score");
my $dashes = $header;
$dashes =~ s/./-/g;
print $dashes;
print $header;
print $dashes;


my $total_score = 0;

foreach my $scene (@scene_names){
    my $score;
    my $fast_time = $fast_times{$scene};
    my $naive_time = $naive_times{$scene};
    my $time = $time_required{$scene};

    if ($correct{$scene}) {
        if ($time <= 1.20 * $fast_time) {
            $score = $perf_points + $correctness_points;
        }
        elsif ($time > $naive_time) {
            $score = $correctness_points;
        }
        else {
            $score = $correctness_points + ceil ($perf_points * ($fast_time /$time));
        }
    }
    else {
        $time .= " (F)";
        $score = 0;
    }

    printf ("| %-15s | %-15s | %-15s | %-15s | %-15s |\n", "$scene", "$naive_time", "$fast_time", "$time", "$score");
    $total_score += $score;
}
#printf ("%64s %-15s\n", "| Total : |", "$total_score |");
print $dashes;
printf ("  %-15s   %-15s   %-15s | %-15s | %-15s |\n", "", "", "", "Total score:", 
    $total_score . "/" . ($perf_points+$correctness_points) * ($#scene_names + 1));
print $dashes;

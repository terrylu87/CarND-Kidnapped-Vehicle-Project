/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 20;
    default_random_engine gen;

    // Creates a normal (Gaussian) distribution for x,y,theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle s;
        s.id=i;
        s.x = dist_x(gen);
        s.y = dist_y(gen);
        s.theta = dist_theta(gen);
        s.weight = 1;
        particles.push_back(s);
        weights.push_back(s.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    // Create normal distributions x for y and theta
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for(int i=0;i<num_particles;++i){
        double x0 = particles[i].x;
        double y0 = particles[i].y;
        double t0 = particles[i].theta;
        double xf,yf,tf;
        if(fabs(yaw_rate) > 0.0001){
            tf = t0 + yaw_rate*delta_t;
            xf = x0 + velocity/yaw_rate*(sin(tf)-sin(t0));
            yf = y0 + velocity/yaw_rate*(cos(t0)-cos(tf));
        }else{
            tf = t0;
            xf = x0 + velocity*cos(t0)*delta_t;
            yf = y0 + velocity*sin(t0)*delta_t;
        }
        particles[i].x = xf + dist_x(gen);
        particles[i].y = yf + dist_y(gen);
        particles[i].theta = tf + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> landmark_list,std::vector<unsigned int> valid_list, std::vector<LandmarkObs>& observations) {
    // Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.

    for(unsigned int i=0;i<observations.size();++i)
    {
        double shortest_dist=99999;
        int idx = -1;
        for(unsigned int j=0;j<valid_list.size();++j)
        {
            double distance = dist(landmark_list[valid_list[j]].x_f,
                                   landmark_list[valid_list[j]].y_f,
                                   observations[i].x,
                                   observations[i].y);
            if(distance < shortest_dist){
                shortest_dist = distance;
                idx = landmark_list[valid_list[j]].id_i;
            }
        }
        observations[i].id = idx;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    if(observations.size()==0){
        return;
    }

    weights.clear();

    const vector<Map::single_landmark_s>& landmark_list = map_landmarks.landmark_list;
    // for each particle:
    unsigned int i;
    for(i=0;i<num_particles;++i)
    {
        // get valid landmarks inside sensor_range
        vector<unsigned int> valid_landmarks;
        //vector<LandmarkObs> valid_landmarks;
        unsigned int j;

        for(j=0;j<landmark_list.size();++j)
        {
            if(dist(particles[i].x,particles[i].y,landmark_list[j].x_f,landmark_list[j].y_f)
               < sensor_range){
                valid_landmarks.push_back(j);
            }
        }
        // transform observations into map coordinates
        vector<LandmarkObs> obs_map;
        for(j=0;j<observations.size();++j)
        {
            LandmarkObs o;
            double cos_theta = cos(particles[i].theta);
            double sin_theta = sin(particles[i].theta);
            o.x = particles[i].x + cos_theta*observations[j].x - sin_theta*observations[j].y;
            o.y = particles[i].y + sin_theta*observations[j].x + cos_theta*observations[j].y;
            obs_map.push_back(o);
        }
        // assocate the observations and the landmarks
        dataAssociation(landmark_list,valid_landmarks,obs_map);
        SetAssociations(particles[i], obs_map);
        // calcuate the weight

        double x_cft = 1.0 / (2.0 * std_landmark[0] * std_landmark[0]);
        double y_cft = 1.0 / (2.0 * std_landmark[1] * std_landmark[1]);
        double xy_cft = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

        double prob = 1.0;
        for(j=0;j<obs_map.size();++j)
        {
            double tx=x_cft*pow((obs_map[j].x-landmark_list[obs_map[j].id-1].x_f),2);
            double ty=y_cft*pow((obs_map[j].y-landmark_list[obs_map[j].id-1].y_f),2);
            double p = xy_cft * exp(-1.0*(tx+ty));
            prob *= p;
        }
        particles[i].weight = prob;
        weights.push_back(prob);
    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::default_random_engine gen;
    std::discrete_distribution<int> dist(weights.begin(), weights.end());

    std::vector<Particle> resample_p;

    for(int i = 0; i < num_particles; i++) {
        int index = dist(gen);
        resample_p.push_back(particles.at(index));
    }

    particles = resample_p;
}

void ParticleFilter::SetAssociations(Particle& particle, const vector<LandmarkObs>& obs_map)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    for(int j=0;j<obs_map.size();++j) {
        particle.associations.push_back(obs_map[j].id);
        particle.sense_x.push_back(obs_map[j].x);
        particle.sense_y.push_back(obs_map[j].y);
    }
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

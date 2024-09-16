
# Abstract

Der Abstract wird grundsätzlich als letztes geschrieben. Die Überprüfbarkeit der Arbeit (=letzter Teil des Abstracts) ergibt sich aus den Ergebnissen, diese gibt es ja noch nicht. Hier ist vorläufiges Feedback zum Abstract.

"The goal of this work is to develop a system that enables a mobile robot to navigate optimally and collision-free."


"Several established approaches,such as the Dynamic Window Approach (DWA) and Move Base Flex, have already proven effective. The Dynamic Window Approach limits the robot's velocity space to ensure collision avoidance, while Move Base Flex provides a flexible interface for path planning and execution, allowing for the integration of different navigation frameworks."

Hier fehlt ein "aber" (siehe Problem description arising from motivation). Hier muss stehen, was der Sota nicht kann

"In this work, a control system is developed for dynamic obstacle avoidance using a nonlinear Model Predictive Controller (nMPC)."

Das gehort nach oben zu "The goal of this work..."

"The results are compared with the well-established Move Base DWA from ROS1. Both controllers are evaluated in a Gazebo simulation under three conditions: with static objects, with dynamically detected obstacles, and without obstacle avoidance (i.e., without laser scan)."

Hier präzisieren, was/wie genau evaluiert wird und welche Ergebnisse erzielt wurden (=Überprüfbarkeit). Das ist aber von den konkreten Ergebnissen abhängig.


# Introduction

Folgende Punkte sind laut IMRAD Teil von Ch1:

* Introduction to the paper’s main subject (subject = lokale Pfadplanung)
* Motivation for creating paper contribution (sehr grobe Übersicht über sota)
* Problem description arising from motivation (was fehlt im aktuellen sota -> vermeidung dynamischer Objekte)
* Paper contribution
* Paper structure

### Wiss. Beitrag (contribution)

"This work contributes to the development of a nonlinear Model Predictive Controller (nMPC)
for a differential drive mobile robot with integrated dynamic obstacle avoidance."

Hier fehlt was beigetragen wird. Präziser an den SOTA anknüpfen und sich konkret auf den einen weiterentwickelten Aspekt Ihrer Lösung fokussieren. Basierend auf Abstract und der restlichen Contribution ist das die Vermeidung dynamischer Objekte, aber es muss durch aktuelle Quellen (= Konferenzen/Journals der letzten 5 Jahre) begründet werde, dass es den Beitrag noch nicht in dieser Form gibt.

"The nMPC controller is compared to the widely used Dynamic Window Approach (DWA) from the ROS Navigation stack."

Widely used ist nicht wirklich quantifizierbar. Hier schreibt man in Papern stattdessen "seminal work". Seminal Work sind arbeiten, die nicht (mehr) Stand der Technik sind, aber durch ihre Robustheit und generelle Anwendbarkeit vermehrt eingesetzt werden. Hier auch spezifizieren, wie gegen Seminal Work und den State of the Art evaluiert wird.

"The main objective is to design a controller that effectively addresses the challenges of dynamic obstacle avoidance, offering improved performance in real-time navigation within dynamic environments."

Das würde ich nach den ersten Satz schieben (bezieht sich ja darauf)


# State of the Art

Move Base DWA Planner
ROS MPC Local Planner
Move Base Flex Paper

Grundsätzlich in Ordnung so, aber alle diese Quellen sind seminal Work. Hier fehlen aktuelle Quellen aus der Forschung


# Methods

"nMPC
Trajectory Planner
Obstacle Detection"

Bitte hier auch das verwendete Modell beschreiben/herleiten.


# Implementation
  
"CasAdi
ROS Nav
Parameter Settings for nMPC controller
    Table with all params for controller, trajectory planner and obstacle detection"


Besserer Name: Implementation Detail (ist bei Papern so üblich)


# Results

In Ordnung so, aber Evaluierungsmetriken bitte vor den Experimenten einführen (stehen aktuell nachher)

# Discussion

Muss nicht unbedingt ein eigenes Kapitel sein. Falls es im Paper Platzprobleme gibt, wird Discussion auch gerne in die Resultate direkt integriert.
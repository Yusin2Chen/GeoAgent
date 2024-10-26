# GeoAgent
GeoAgent is an open-source framework designed to facilitate Earth Observation (EO) tasks by leveraging state-of-the-art deep learning and machine learning tools. GeoAgent provides an accessible and scalable solution for researchers and practitioners working with EO data, integrating diverse data sources, models, and libraries to support tasks ranging from land cover classification to high-resolution image segmentation.

## Key Features
### Data Acquisition Tools:
GeoAgent integrates with various open-source EO data sources in Google Earth Engine (GEE), enabling the retrieval and processing of data from the Sentinel-1 and Sentinel-2 satellites, Dynamic Land Cover, and more. This setup allows GeoAgent to handle open-domain EO tasks with ease.

### Deep Learning Tools:
GeoAgent incorporates popular deep learning libraries like terratorch and torchgeo, providing built-in support for datasets and models that can be trained with default configurations. These libraries also support domain-specific geospatial foundation models from IBM, including the NASA-IBM GFM Prithvi models. Currently, the framework includes Sentinel-1 and Sentinel-2-based deep learning models for key tasks:
Multitemporal Crop Classification
Burn Scar Segmentation
Forest Degradation Segmentation
Flood Mapping
### High-Resolution RGB Imaging Tools:
For local studies requiring high-resolution RGB images, GeoAgent includes universal object detection and segmentation tools like samgeo and efficient-yolo-world. These tools enhance GeoAgent's capacity to perform detailed image analysis and segmentation for a variety of applications.

### Machine Learning Tools:
Although deep learning models form the core of GeoAgent, traditional machine learning tools are included as well, adding flexibility for tasks where simpler or interpretable models are beneficial.

## Getting Started
To get started with GeoAgent, clone the repository and follow the installation steps for the necessary dependencies. Refer to the examples folder for detailed use cases and sample code.

#TODO
### Install dependencies (see requirements.txt)
#TODO
pip install -r requirements.txt
Prerequisites
Python 3.8+
Google Earth Engine API access
Deep learning libraries: terratorch, torchgeo
Object detection and segmentation tools: samgeo, efficient-yolo-world

## Usage
Once installed, GeoAgent provides an interface to access and process EO data, #TODO.

## Contributing
Contributions are welcome! Please see our contributing guidelines for details on how to contribute to this project.

## License
GeoAgent is licensed under the MIT License. See LICENSE for more information.

## Acknowledgement
Thanks partial code for [the help from ](https://github.com/NirantK/agentai)

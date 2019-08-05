# @author  Zhang Handuo (hzhang032@e.ntu.edu.sg)

# first we can indicate the documentation build as an option and set it to ON by default
option(BUILD_DOC "Build documentation" ON)

if(BUILD_DOC)
    # check if Doxygen is installed
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        # set input and output files
        set(DOXYGEN_IN Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
        set(DOXYGEN_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/docs)
        # request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message("Doxygen build started")

        # note the option ALL which allows to build the doc together with the application
        add_custom_target( doc_doxygen_cubicle ALL
                COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Generating API documentation with Doxygen"
                VERBATIM )
    else (DOXYGEN_FOUND)
        message("Doxygen need to be installed to generate the doxygen documentation")
    endif (DOXYGEN_FOUND)

    find_package(Sphinx REQUIRED)

    # HTML output directory
#    set(SPHINX_HTML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs/sphinx/html")
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
    set(SPHINX_BUILD ${PROJECT_SOURCE_DIR}/docs/sphinx)

    # Building using 'make_docs.py' python script
    add_custom_target(sslam_doc_cubicle ALL
            ${SPHINX_EXECUTABLE} -b html
            ${SPHINX_SOURCE} ${SPHINX_BUILD}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Building HTML documentation with Sphinx")
endif()
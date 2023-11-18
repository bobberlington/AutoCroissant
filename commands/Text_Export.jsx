/*****************************************************************
 *
 * TextConvert.Export 1.1 - by Bramus! - https://www.bram.us/
 *
 * v 1.1 - 2016.02.17 - UTF-8 support
 *                      Update license to MIT License
 *
 * v 1.0 - 2008.10.30 - (based upon TextExport 1.3, without the "save dialog" option)
 *
 *****************************************************************
 *
 * Copyright (c) 2016 Bram(us) Van Damme - https://www.bram.us/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *****************************************************************/

	/**
	 *  TextConvert.Export Init function
	 * -------------------------------------------------------------
	 */

write_stats_at_end = false;
last_path = "";
hp = "";
def = "";
atk = "";
spd = "";

	 	function initTextConvertExport() {

			// Linefeed shizzle
			if ($.os.search(/windows/i) != -1)
				fileLineFeed = "windows";
			else
				fileLineFeed = "macintosh";

			// Do we have a document open?
			if (app.documents.length === 0) {
				alert("Please open a file", "TextConvert.Export Error", true);
				return;
			}

			// Oh, we have more than one document open!
			if (app.documents.length > 1) {

				var runMultiple = confirm("TextConvert.Export has detected Multiple Files.\nDo you wish to run TextConvert.Export on all opened files?", true, "TextConvert.Export");

				if (runMultiple === true) {
					docs	= app.documents;
				} else {
					docs	= [app.activeDocument];
				}

			// Only one document open
			} else {

				runMultiple 	= false;
				docs 			= [app.activeDocument];

			}

			// Loop all documents
			for (var i = 0; i < docs.length; i++)
			{

				// Auto set filePath and fileName
				filePath = Folder.myDocuments + '/descriptions/' + docs[i].name + '.txt';

				// create outfile
				var fileOut	= new File(filePath);

				// set linefeed
				fileOut.linefeed = fileLineFeed;

				// Set encoding
				fileOut.encoding = "UTF8"

				// open for write
				fileOut.open("w", "TEXT", "????");

				// Set active document
				app.activeDocument = docs[i];

				write_stats_at_end = false;
				last_path = "";
				hp = "";
				def = "";
				atk = "";
				spd = "";
				// call to the core with the current document
				goTextExport2(app.activeDocument, fileOut, '/');

				if (hp == "-1")
					fileOut.writeln('[ hp=10 ]');
				else if (write_stats_at_end)
					fileOut.writeln('[ hp=' + hp + ' ]');
				if (def == "-1")
					fileOut.writeln('[ def=10 ]');
				else if (write_stats_at_end)
					fileOut.writeln('[ def=' + def + ' ]');
				if (atk == "-1")
					fileOut.writeln('[ atk=10 ]');
				else if (write_stats_at_end)
					fileOut.writeln('[ atk=' + atk + ' ]');
				if (spd == "-1")
					fileOut.writeln('[ spd=10 ]');
				else if (write_stats_at_end)
					fileOut.writeln('[ spd=' + spd + ' ]');

				// close the file
				fileOut.close();

			}

			// Post processing: give notice (multiple) or open file (single)
			if (runMultiple === true) {
				alert("Parsed " + documents.length + " files;\nFiles were saved in your documents folder", "TextExport");
			} else {
				fileOut.execute();
			}

		}


  	/**
  	 * TextExport Core Function (V2)
  	 * -------------------------------------------------------------
	 */

		function goTextExport2(el, fileOut, path)
		{

			// Get the layers
			var layers = el.layers;

			// Loop 'm
			for (var layerIndex = layers.length; layerIndex > 0; layerIndex--)
			{

				// curentLayer ref
				var currentLayer = layers[layerIndex-1];

				// currentLayer is a LayerSet
				if (currentLayer.typename == "LayerSet") {

					goTextExport2(currentLayer, fileOut, path + currentLayer.name + '/');

				// currentLayer is not a LayerSet
				} else {

					// Layer is visible and Text --> we can haz copy paste!
					//  && (currentLayer.kind == LayerKind.TEXT)
					if ( (currentLayer.visible) ) {
						if (path.toLowerCase().indexOf("bars/dark") !== -1) {
							if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("hp") !== -1) {
								fileOut.writeln('[ hp=' + currentLayer.name + ' ]');
								hp = currentLayer.name;
							} else if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("def") !== -1) {
								fileOut.writeln('[ def=' + currentLayer.name + ' ]');
								def = currentLayer.name;
							} else if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("atk") !== -1) {
								fileOut.writeln('[ atk=' + currentLayer.name + ' ]');
								atk = currentLayer.name;
							} else if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("spd") !== -1) {
								fileOut.writeln('[ spd=' + currentLayer.name + ' ]');
								spd = currentLayer.name;
							}
						} else if (path.toLowerCase().indexOf("bars/hp") !== -1 || path.toLowerCase().indexOf("bars/def") !== -1 || path.toLowerCase().indexOf("bars/atk") !== -1 || path.toLowerCase().indexOf("bars/spd") !== -1) {
							write_stats_at_end = true;
							if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("hp") !== -1) {
								hp = currentLayer.name;
							} else if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("def") !== -1) {
								def = currentLayer.name;
							} else if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("atk") !== -1) {
								atk = currentLayer.name;
							} else if (!isNaN(currentLayer.name) && path.toLowerCase().indexOf("spd") !== -1) {
								spd = currentLayer.name;
							}
						} else if (currentLayer.name.toLowerCase().indexOf("name") !== -1 || path.toLowerCase().indexOf("stat") !== -1 || path.toLowerCase().indexOf("bar") !== -1 || path.toLowerCase().indexOf("line") !== -1 || path.toLowerCase().indexOf("image") !== -1) {
							continue;
						} else
							fileOut.writeln('[ ' + currentLayer.name + ' ]');
						if (currentLayer.kind == LayerKind.TEXT)
							fileOut.writeln(currentLayer.textItem.contents);
					} else {
						if (!isNaN(currentLayer.name) && hp == "10" && path.toLowerCase().indexOf("hp") !== -1)
							hp = "0";
						else if (!isNaN(currentLayer.name) && def == "10" && path.toLowerCase().indexOf("def") !== -1)
							def = "0";
						else if (!isNaN(currentLayer.name) && atk == "10" && path.toLowerCase().indexOf("atk") !== -1)
							atk = "0";
						else if (!isNaN(currentLayer.name) && spd == "10" && path.toLowerCase().indexOf("spd") !== -1)
							spd = "0";
						else if (!hp && path.toLowerCase().indexOf("hp") !== -1)
							hp = "-1";
						else if (!def && path.toLowerCase().indexOf("def") !== -1)
							def = "-1";
						else if (!atk && path.toLowerCase().indexOf("atk") !== -1)
							atk = "-1";
						else if (!spd && path.toLowerCase().indexOf("spd") !== -1)
							spd = "-1";
					}
				}
				last_path = path;
			}
		}


	/**
	 *  TextConvert.Export Boot her up
	 * -------------------------------------------------------------
	 */

	 	initTextConvertExport();
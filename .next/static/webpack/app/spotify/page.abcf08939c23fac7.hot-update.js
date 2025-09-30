/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
self["webpackHotUpdate_N_E"]("app/spotify/page",{

/***/ "(app-pages-browser)/./utils/reportGenerator.js":
/*!**********************************!*\
  !*** ./utils/reportGenerator.js ***!
  \**********************************/
/***/ ((module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval(__webpack_require__.ts("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   generateReport: () => (/* binding */ generateReport)\n/* harmony export */ });\n/* harmony import */ var jspdf__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! jspdf */ \"(app-pages-browser)/./node_modules/jspdf/dist/jspdf.es.min.js\");\n/* harmony import */ var _reportTemplates__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./reportTemplates */ \"(app-pages-browser)/./utils/reportTemplates.js\");\n/* harmony import */ var _reportTemplates__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_reportTemplates__WEBPACK_IMPORTED_MODULE_1__);\n\n\nfunction generateReport(platform) {\n    const doc = new jspdf__WEBPACK_IMPORTED_MODULE_0__.jsPDF();\n    const date = new Date().toLocaleString();\n    doc.setFontSize(18);\n    doc.text(\"\".concat(platform, \" Dashboard Report\"), 10, 20);\n    doc.setFontSize(12);\n    doc.text(\"Generated on: \".concat(date), 10, 30);\n    switch(platform.toLowerCase()){\n        case \"youtube\":\n            (0,_reportTemplates__WEBPACK_IMPORTED_MODULE_1__.addYouTubeReport)(doc);\n            break;\n        case \"facebook\":\n            (0,_reportTemplates__WEBPACK_IMPORTED_MODULE_1__.addFacebookReport)(doc);\n            break;\n        case \"spotify\":\n            (0,_reportTemplates__WEBPACK_IMPORTED_MODULE_1__.addSpotifyReport)(doc);\n            break;\n        // ...other dashboards\n        default:\n            doc.text(\"No template available for this platform.\", 10, 50);\n    }\n    doc.save(\"\".concat(platform, \"-report.pdf\"));\n}\n\n\n;\n    // Wrapped in an IIFE to avoid polluting the global scope\n    ;\n    (function () {\n        var _a, _b;\n        // Legacy CSS implementations will `eval` browser code in a Node.js context\n        // to extract CSS. For backwards compatibility, we need to check we're in a\n        // browser context before continuing.\n        if (typeof self !== 'undefined' &&\n            // AMP / No-JS mode does not inject these helpers:\n            '$RefreshHelpers$' in self) {\n            // @ts-ignore __webpack_module__ is global\n            var currentExports = module.exports;\n            // @ts-ignore __webpack_module__ is global\n            var prevSignature = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevSignature) !== null && _b !== void 0 ? _b : null;\n            // This cannot happen in MainTemplate because the exports mismatch between\n            // templating and execution.\n            self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n            // A module can be accepted automatically based on its exports, e.g. when\n            // it is a Refresh Boundary.\n            if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n                // Save the previous exports signature on update so we can compare the boundary\n                // signatures. We avoid saving exports themselves since it causes memory leaks (https://github.com/vercel/next.js/pull/53797)\n                module.hot.dispose(function (data) {\n                    data.prevSignature =\n                        self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports);\n                });\n                // Unconditionally accept an update to this module, we'll check if it's\n                // still a Refresh Boundary later.\n                // @ts-ignore importMeta is replaced in the loader\n                module.hot.accept();\n                // This field is set when the previous version of this module was a\n                // Refresh Boundary, letting us know we need to check for invalidation or\n                // enqueue an update.\n                if (prevSignature !== null) {\n                    // A boundary can become ineligible if its exports are incompatible\n                    // with the previous exports.\n                    //\n                    // For example, if you add/remove/change exports, we'll want to\n                    // re-execute the importing modules, and force those components to\n                    // re-render. Similarly, if you convert a class component to a\n                    // function, we want to invalidate the boundary.\n                    if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevSignature, self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports))) {\n                        module.hot.invalidate();\n                    }\n                    else {\n                        self.$RefreshHelpers$.scheduleUpdate();\n                    }\n                }\n            }\n            else {\n                // Since we just executed the code for the module, it's possible that the\n                // new exports made it ineligible for being a boundary.\n                // We only care about the case when we were _previously_ a boundary,\n                // because we already accepted this update (accidental side effect).\n                var isNoLongerABoundary = prevSignature !== null;\n                if (isNoLongerABoundary) {\n                    module.hot.invalidate();\n                }\n            }\n        }\n    })();\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwcC1wYWdlcy1icm93c2VyKS8uL3V0aWxzL3JlcG9ydEdlbmVyYXRvci5qcyIsIm1hcHBpbmdzIjoiOzs7Ozs7O0FBQThCO0FBQzREO0FBRW5GLFNBQVNJLGVBQWVDLFFBQVE7SUFDckMsTUFBTUMsTUFBTSxJQUFJTix3Q0FBS0E7SUFDckIsTUFBTU8sT0FBTyxJQUFJQyxPQUFPQyxjQUFjO0lBRXRDSCxJQUFJSSxXQUFXLENBQUM7SUFDaEJKLElBQUlLLElBQUksQ0FBQyxHQUFZLE9BQVROLFVBQVMsc0JBQW9CLElBQUk7SUFDN0NDLElBQUlJLFdBQVcsQ0FBQztJQUNoQkosSUFBSUssSUFBSSxDQUFDLGlCQUFzQixPQUFMSixPQUFRLElBQUk7SUFFdEMsT0FBUUYsU0FBU08sV0FBVztRQUMxQixLQUFLO1lBQ0hYLGtFQUFnQkEsQ0FBQ0s7WUFDakI7UUFDRixLQUFLO1lBQ0hKLG1FQUFpQkEsQ0FBQ0k7WUFDbEI7UUFDRixLQUFLO1lBQ0hILGtFQUFnQkEsQ0FBQ0c7WUFDakI7UUFDRixzQkFBc0I7UUFDdEI7WUFDRUEsSUFBSUssSUFBSSxDQUFDLDRDQUE0QyxJQUFJO0lBQzdEO0lBRUFMLElBQUlPLElBQUksQ0FBQyxHQUFZLE9BQVRSLFVBQVM7QUFDdkIiLCJzb3VyY2VzIjpbIkM6XFxVc2Vyc1xcQWN1bmluXFxEb2N1bWVudHNcXEdpdEh1YlxcU3VnYXJjYW5lX0FydGlzdF9NYW5hZ2VtZW50XFx1dGlsc1xccmVwb3J0R2VuZXJhdG9yLmpzIl0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IGpzUERGIH0gZnJvbSBcImpzcGRmXCI7XHJcbmltcG9ydCB7IGFkZFlvdVR1YmVSZXBvcnQsIGFkZEZhY2Vib29rUmVwb3J0LCBhZGRTcG90aWZ5UmVwb3J0IH0gZnJvbSBcIi4vcmVwb3J0VGVtcGxhdGVzXCI7XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gZ2VuZXJhdGVSZXBvcnQocGxhdGZvcm0pIHtcclxuICBjb25zdCBkb2MgPSBuZXcganNQREYoKTtcclxuICBjb25zdCBkYXRlID0gbmV3IERhdGUoKS50b0xvY2FsZVN0cmluZygpO1xyXG5cclxuICBkb2Muc2V0Rm9udFNpemUoMTgpO1xyXG4gIGRvYy50ZXh0KGAke3BsYXRmb3JtfSBEYXNoYm9hcmQgUmVwb3J0YCwgMTAsIDIwKTtcclxuICBkb2Muc2V0Rm9udFNpemUoMTIpO1xyXG4gIGRvYy50ZXh0KGBHZW5lcmF0ZWQgb246ICR7ZGF0ZX1gLCAxMCwgMzApO1xyXG5cclxuICBzd2l0Y2ggKHBsYXRmb3JtLnRvTG93ZXJDYXNlKCkpIHtcclxuICAgIGNhc2UgXCJ5b3V0dWJlXCI6XHJcbiAgICAgIGFkZFlvdVR1YmVSZXBvcnQoZG9jKTtcclxuICAgICAgYnJlYWs7XHJcbiAgICBjYXNlIFwiZmFjZWJvb2tcIjpcclxuICAgICAgYWRkRmFjZWJvb2tSZXBvcnQoZG9jKTtcclxuICAgICAgYnJlYWs7XHJcbiAgICBjYXNlIFwic3BvdGlmeVwiOlxyXG4gICAgICBhZGRTcG90aWZ5UmVwb3J0KGRvYyk7XHJcbiAgICAgIGJyZWFrO1xyXG4gICAgLy8gLi4ub3RoZXIgZGFzaGJvYXJkc1xyXG4gICAgZGVmYXVsdDpcclxuICAgICAgZG9jLnRleHQoXCJObyB0ZW1wbGF0ZSBhdmFpbGFibGUgZm9yIHRoaXMgcGxhdGZvcm0uXCIsIDEwLCA1MCk7XHJcbiAgfVxyXG5cclxuICBkb2Muc2F2ZShgJHtwbGF0Zm9ybX0tcmVwb3J0LnBkZmApO1xyXG59XHJcbiJdLCJuYW1lcyI6WyJqc1BERiIsImFkZFlvdVR1YmVSZXBvcnQiLCJhZGRGYWNlYm9va1JlcG9ydCIsImFkZFNwb3RpZnlSZXBvcnQiLCJnZW5lcmF0ZVJlcG9ydCIsInBsYXRmb3JtIiwiZG9jIiwiZGF0ZSIsIkRhdGUiLCJ0b0xvY2FsZVN0cmluZyIsInNldEZvbnRTaXplIiwidGV4dCIsInRvTG93ZXJDYXNlIiwic2F2ZSJdLCJpZ25vcmVMaXN0IjpbXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///(app-pages-browser)/./utils/reportGenerator.js\n"));

/***/ }),

/***/ "(app-pages-browser)/./utils/reportTemplates.js":
/*!**********************************!*\
  !*** ./utils/reportTemplates.js ***!
  \**********************************/
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {



;
    // Wrapped in an IIFE to avoid polluting the global scope
    ;
    (function () {
        var _a, _b;
        // Legacy CSS implementations will `eval` browser code in a Node.js context
        // to extract CSS. For backwards compatibility, we need to check we're in a
        // browser context before continuing.
        if (typeof self !== 'undefined' &&
            // AMP / No-JS mode does not inject these helpers:
            '$RefreshHelpers$' in self) {
            // @ts-ignore __webpack_module__ is global
            var currentExports = module.exports;
            // @ts-ignore __webpack_module__ is global
            var prevSignature = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevSignature) !== null && _b !== void 0 ? _b : null;
            // This cannot happen in MainTemplate because the exports mismatch between
            // templating and execution.
            self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);
            // A module can be accepted automatically based on its exports, e.g. when
            // it is a Refresh Boundary.
            if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
                // Save the previous exports signature on update so we can compare the boundary
                // signatures. We avoid saving exports themselves since it causes memory leaks (https://github.com/vercel/next.js/pull/53797)
                module.hot.dispose(function (data) {
                    data.prevSignature =
                        self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports);
                });
                // Unconditionally accept an update to this module, we'll check if it's
                // still a Refresh Boundary later.
                // @ts-ignore importMeta is replaced in the loader
                module.hot.accept();
                // This field is set when the previous version of this module was a
                // Refresh Boundary, letting us know we need to check for invalidation or
                // enqueue an update.
                if (prevSignature !== null) {
                    // A boundary can become ineligible if its exports are incompatible
                    // with the previous exports.
                    //
                    // For example, if you add/remove/change exports, we'll want to
                    // re-execute the importing modules, and force those components to
                    // re-render. Similarly, if you convert a class component to a
                    // function, we want to invalidate the boundary.
                    if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevSignature, self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports))) {
                        module.hot.invalidate();
                    }
                    else {
                        self.$RefreshHelpers$.scheduleUpdate();
                    }
                }
            }
            else {
                // Since we just executed the code for the module, it's possible that the
                // new exports made it ineligible for being a boundary.
                // We only care about the case when we were _previously_ a boundary,
                // because we already accepted this update (accidental side effect).
                var isNoLongerABoundary = prevSignature !== null;
                if (isNoLongerABoundary) {
                    module.hot.invalidate();
                }
            }
        }
    })();


/***/ })

});